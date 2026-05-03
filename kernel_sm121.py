"""
kernel_sm121.py — Optimized kernels for DeepSeek V4 Flash on sm_121 (DGX Spark).

Drop-in replacement for the official TileLang-based kernel.py.
All 6 functions have identical signatures.

- fp4_gemm / fp8_gemm: Triton fused kernels (dequant + scale + matmul in one pass)
- sparse_attn: Vectorized PyTorch (no Python loops)
- act_quant / fp4_act_quant / hc_split_sinkhorn: Pure PyTorch (already fast enough)

Usage: in model.py, change
    from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn
to:
    from kernel_sm121 import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn
"""

import torch
import torch.nn.functional as F
from typing import Optional
from collections import defaultdict
import time as _time

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# Lightweight profiler: accumulates wall-clock time per kernel
_PROFILE = defaultdict(lambda: [0, 0.0])  # {name: [count, total_seconds]}

def profile_report():
    """Print accumulated kernel timing stats and reset."""
    if not _PROFILE:
        return
    print("[kernel_sm121] === Profile Report ===", flush=True)
    total = sum(v[1] for v in _PROFILE.values())
    for name, (cnt, sec) in sorted(_PROFILE.items(), key=lambda x: -x[1][1]):
        pct = sec / total * 100 if total > 0 else 0
        print(f"  {name:25s}: {cnt:6d} calls, {sec:7.3f}s ({pct:5.1f}%)", flush=True)
    print(f"  {'TOTAL':25s}: {sum(v[0] for v in _PROFILE.values()):6d} calls, {total:7.3f}s", flush=True)
    _PROFILE.clear()


# ============================================================
# 1. act_quant — Block-wise FP8 quantization (pure PyTorch, fast enough)
# ============================================================

def _round_scale_pow2(scale: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(scale)))


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32, inplace: bool = False,
) -> torch.Tensor:
    _t0 = _time.perf_counter()
    orig_shape = x.shape
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous().float()
    flat = z.view(-1, N)
    M = flat.size(0)
    n_blocks = N // block_size
    blocks = flat.view(M, n_blocks, block_size)
    amax = blocks.abs().amax(dim=-1).clamp(min=1e-4)
    if scale_fmt is not None:
        scale = _round_scale_pow2(amax / 448.0)
    else:
        scale = amax / 448.0
    scaled = blocks / scale.unsqueeze(-1)
    clamped = scaled.clamp(-448.0, 448.0)
    if inplace:
        quantized = clamped.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)
        result = quantized.view(orig_shape).to(x.dtype)
        x.copy_(result)
        _PROFILE['act_quant_inplace'][0] += 1
        _PROFILE['act_quant_inplace'][1] += _time.perf_counter() - _t0
        return x
    else:
        y = clamped.view(M, N).to(torch.float8_e4m3fn).view(*orig_shape[:-1], N)
        s = scale.to(scale_dtype).view(*orig_shape[:-1], n_blocks)
        _PROFILE['act_quant'][0] += 1
        _PROFILE['act_quant'][1] += _time.perf_counter() - _t0
        return y, s


# ============================================================
# 2. fp4_act_quant — Block-wise FP4 simulation quantization (pure PyTorch)
# ============================================================

FP4_MAX = 6.0

def fp4_act_quant(
    x: torch.Tensor, block_size: int = 32, inplace: bool = False,
) -> torch.Tensor:
    orig_shape = x.shape
    N = x.size(-1)
    assert N % block_size == 0
    z = x.contiguous().float()
    flat = z.view(-1, N)
    M = flat.size(0)
    n_blocks = N // block_size
    blocks = flat.view(M, n_blocks, block_size)
    amax = blocks.abs().amax(dim=-1).clamp(min=FP4_MAX * (2**-126))
    scale = _round_scale_pow2(amax / FP4_MAX)
    scaled = blocks / scale.unsqueeze(-1)
    clamped = scaled.clamp(-FP4_MAX, FP4_MAX)
    if inplace:
        fp4_values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=torch.float32
        )
        signs = clamped.sign()
        abs_vals = clamped.abs()
        diffs = (abs_vals.unsqueeze(-1) - fp4_values).abs()
        nearest_idx = diffs.argmin(dim=-1)
        quantized = signs * fp4_values[nearest_idx]
        result = (quantized * scale.unsqueeze(-1)).view(orig_shape).to(x.dtype)
        x.copy_(result)
        return x
    else:
        raise NotImplementedError("fp4_act_quant non-inplace not needed for sm_121 fallback")


# ============================================================
# Shared: E8M0 scale conversion
# ============================================================

def _e8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 (float8_e8m0fnu) scale to float32: 2^(e-127)."""
    return (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)


# ============================================================
# 3. fp8_gemm — Triton fused FP8×FP8 GEMM with per-block scaling
# ============================================================

if HAS_TRITON:
    @triton.jit
    def _fp8_gemm_kernel(
        a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, out_ptr,
        M, N, K: tl.constexpr,
        a_stride_m, a_stride_k,
        as_stride_m, as_stride_kb,
        b_stride_n, b_stride_k,
        bs_stride_nb, bs_stride_kb,
        out_stride_m, out_stride_n,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            kb = k0 // BLOCK_K

            a = tl.load(
                a_ptr + offs_m[:, None] * a_stride_m + k[None, :] * a_stride_k,
                mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0,
            )
            b = tl.load(
                b_ptr + offs_n[:, None] * b_stride_n + k[None, :] * b_stride_k,
                mask=(offs_n[:, None] < N) & (k[None, :] < K), other=0.0,
            )
            a_s = tl.load(
                a_scale_ptr + offs_m * as_stride_m + kb * as_stride_kb,
                mask=offs_m < M, other=0.0,
            ).to(tl.float32)
            b_s = tl.load(
                b_scale_ptr + (offs_n // BLOCK_K) * bs_stride_nb + kb * bs_stride_kb,
                mask=offs_n < N, other=0.0,
            ).to(tl.float32)

            acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32) * a_s[:, None] * b_s[None, :]

        tl.store(
            out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp8[N,K]^T with per-128 block scaling."""
    _t0 = _time.perf_counter()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    out_shape = (*a.shape[:-1], N)

    a_flat = a.view(M, K).contiguous()
    b_flat = b.contiguous()

    a_s_flat = a_s.view(M, -1)
    if a_s_flat.dtype == torch.float8_e8m0fnu:
        a_s_flat = _e8m0_to_fp32(a_s_flat)
    else:
        a_s_flat = a_s_flat.float()
    a_s_flat = a_s_flat.contiguous()

    b_s_flat = b_s
    if b_s_flat.dtype == torch.float8_e8m0fnu:
        b_s_flat = _e8m0_to_fp32(b_s_flat)
    else:
        b_s_flat = b_s_flat.float()
    b_s_flat = b_s_flat.contiguous()

    out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)

    if HAS_TRITON and a.is_cuda:
        BLOCK_K = 128
        BLOCK_M = 16
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _fp8_gemm_kernel[grid](
            a_flat, a_s_flat, b_flat, b_s_flat, out,
            M, N, K,
            a_flat.stride(0), a_flat.stride(1),
            a_s_flat.stride(0), a_s_flat.stride(1),
            b_flat.stride(0), b_flat.stride(1),
            b_s_flat.stride(0), b_s_flat.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
    else:
        block_size = 128
        n_blocks_k = K // block_size
        a_deq = a_flat.float().view(M, n_blocks_k, block_size) * a_s_flat.unsqueeze(-1)
        a_deq = a_deq.view(M, K).to(torch.bfloat16)
        b_deq_parts = []
        n_blocks_n = b_s_flat.size(0)
        for i in range(n_blocks_n):
            start = i * 128
            end = min(start + 128, N)
            rows = b_flat[start:end].float().view(end - start, n_blocks_k, block_size)
            row_scale = b_s_flat[i].unsqueeze(-1).expand(n_blocks_k, block_size)
            rows = rows * row_scale.unsqueeze(0)
            b_deq_parts.append(rows.view(end - start, K))
        b_deq = torch.cat(b_deq_parts, dim=0).to(torch.bfloat16)
        out = torch.mm(a_deq, b_deq.t())

    _PROFILE['fp8_gemm'][0] += 1
    _PROFILE['fp8_gemm'][1] += _time.perf_counter() - _t0
    return out.view(out_shape)


# ============================================================
# 4. fp4_gemm — Triton fused FP8 act × FP4 weight GEMM
# ============================================================

# FP4 E2M1 lookup table (16 entries: low nibble = index)
# Index: sign(1 bit) | exponent(2 bits) | mantissa(1 bit)
FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


if HAS_TRITON:
    @triton.jit
    def _decode_fp4_nibble(nibble):
        """Decode a 4-bit E2M1 nibble to float32.
        nibble layout: [sign(1) | exponent(2) | mantissa(1)]
        Values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}"""
        sign_bit = (nibble >> 3) & 1
        mag_idx = nibble & 0x07
        e = mag_idx >> 1    # 2-bit exponent
        m = mag_idx & 1     # 1-bit mantissa
        is_sub = (e == 0)
        val_norm = (2 + m).to(tl.float32) * tl.exp2((e - 1).to(tl.float32)) * 0.5
        val_sub = m.to(tl.float32) * 0.5
        val = tl.where(is_sub, val_sub, val_norm)
        return tl.where(sign_bit == 1, -val, val)

    @triton.jit
    def _fp4_gemm_kernel(
        a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, out_ptr,
        M, N, K: tl.constexpr,
        a_stride_m, a_stride_k,
        as_stride_m, as_stride_kb,
        b_stride_n, b_stride_kh,
        bs_stride_n, bs_stride_kb,
        out_stride_m, out_stride_n,
        ACT_BLOCK: tl.constexpr,
        W_BLOCK: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """FP8 activation × packed FP4 weight → BF16 output.

        Strategy: iterate K in steps of BLOCK_K (unpacked positions).
        Each step loads BLOCK_K//2 packed bytes from B, unpacks to BLOCK_K floats.
        A is loaded as BLOCK_K FP8 values split into even/odd halves for dot product.

        A: [M, K] fp8_e4m3fn, a_scale: [M, K//ACT_BLOCK] float32
        B: [N, K//2] uint8 (packed FP4), b_scale: [N, K//W_BLOCK] float32
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_kh = tl.arange(0, BLOCK_K // 2)  # half-K offsets for packed B

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            act_kb = k0 // ACT_BLOCK

            # --- Load & dequant A (FP8) ---
            # Even positions: k0, k0+2, k0+4, ...
            a_even_k = k0 + offs_kh * 2
            a_even = tl.load(
                a_ptr + offs_m[:, None] * a_stride_m + a_even_k[None, :] * a_stride_k,
                mask=(offs_m[:, None] < M) & (a_even_k[None, :] < K), other=0.0,
            ).to(tl.float32)
            # Odd positions: k0+1, k0+3, k0+5, ...
            a_odd_k = a_even_k + 1
            a_odd = tl.load(
                a_ptr + offs_m[:, None] * a_stride_m + a_odd_k[None, :] * a_stride_k,
                mask=(offs_m[:, None] < M) & (a_odd_k[None, :] < K), other=0.0,
            ).to(tl.float32)

            # A scale: one scale per ACT_BLOCK elements (all even/odd in same block)
            a_s = tl.load(
                a_scale_ptr + offs_m * as_stride_m + act_kb * as_stride_kb,
                mask=offs_m < M, other=0.0,
            ).to(tl.float32)

            # --- Load & dequant B (packed FP4) ---
            kh = k0 // 2 + offs_kh
            b_packed = tl.load(
                b_ptr + offs_n[:, None] * b_stride_n + kh[None, :] * b_stride_kh,
                mask=(offs_n[:, None] < N) & (kh[None, :] < K // 2), other=0,
            )
            b_low = _decode_fp4_nibble(b_packed & 0x0F)     # even K positions
            b_high = _decode_fp4_nibble((b_packed >> 4) & 0x0F)  # odd K positions

            # B scale: per W_BLOCK (32) elements
            even_wkb = a_even_k // W_BLOCK
            odd_wkb = a_odd_k // W_BLOCK
            bs_even = tl.load(
                b_scale_ptr + offs_n[:, None] * bs_stride_n + even_wkb[None, :] * bs_stride_kb,
                mask=(offs_n[:, None] < N) & (even_wkb[None, :] < K // W_BLOCK), other=1.0,
            ).to(tl.float32)
            bs_odd = tl.load(
                b_scale_ptr + offs_n[:, None] * bs_stride_n + odd_wkb[None, :] * bs_stride_kb,
                mask=(offs_n[:, None] < N) & (odd_wkb[None, :] < K // W_BLOCK), other=1.0,
            ).to(tl.float32)

            b_even_scaled = b_low * bs_even   # [BLOCK_N, BLOCK_K//2]
            b_odd_scaled = b_high * bs_odd    # [BLOCK_N, BLOCK_K//2]

            # Dot: A_even @ B_even^T + A_odd @ B_odd^T, scaled by a_s
            acc += tl.dot(a_even, tl.trans(b_even_scaled), out_dtype=tl.float32) * a_s[:, None]
            acc += tl.dot(a_odd, tl.trans(b_odd_scaled), out_dtype=tl.float32) * a_s[:, None]

        tl.store(
            out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n,
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def _dequant_fp4(b: torch.Tensor) -> torch.Tensor:
    """Dequantize FP4 (float4_e2m1fn_x2) to float32 via int8 view + lookup table."""
    raw = b.view(torch.uint8)
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    table = FP4_TABLE.to(raw.device)
    return torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(-2)


def fp4_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T.
    Uses pre-dequantized B cache for static FP4 weights."""
    _t0 = _time.perf_counter()
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    out_shape = (*a.shape[:-1], N)

    a_flat = a.view(M, K).contiguous()

    a_s_flat = a_s.view(M, -1)
    if a_s_flat.dtype == torch.float8_e8m0fnu:
        a_s_flat = _e8m0_to_fp32(a_s_flat)
    else:
        a_s_flat = a_s_flat.float()
    a_s_flat = a_s_flat.contiguous()

    b_raw = b.view(torch.uint8).contiguous()

    b_s_flat = b_s
    if b_s_flat.dtype == torch.float8_e8m0fnu:
        b_s_flat = _e8m0_to_fp32(b_s_flat)
    else:
        b_s_flat = b_s_flat.float()
    b_s_flat = b_s_flat.contiguous()

    out = torch.empty(M, N, device=a.device, dtype=torch.bfloat16)

    if HAS_TRITON and a.is_cuda:
        BLOCK_K = 128
        BLOCK_M = 16
        BLOCK_N = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _fp4_gemm_kernel[grid](
            a_flat, a_s_flat, b_raw, b_s_flat, out,
            M, N, K,
            a_flat.stride(0), a_flat.stride(1),
            a_s_flat.stride(0), a_s_flat.stride(1),
            b_raw.stride(0), b_raw.stride(1),
            b_s_flat.stride(0), b_s_flat.stride(1),
            out.stride(0), out.stride(1),
            ACT_BLOCK=128, W_BLOCK=32,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4,
        )
    else:
        b_fp32 = _dequant_fp4(b)
        b_s_f = b_s_flat
        weight_block = 32
        n_weight_blocks = K // weight_block
        b_deq = b_fp32.view(N, n_weight_blocks, weight_block) * b_s_f.unsqueeze(-1)
        b_deq = b_deq.view(N, K).to(torch.bfloat16)

        act_block = 128
        n_act_blocks = K // act_block
        a_deq = a_flat.float().view(M, n_act_blocks, act_block) * a_s_flat.unsqueeze(-1)
        a_deq = a_deq.view(M, K).to(torch.bfloat16)
        out = torch.mm(a_deq, b_deq.t())

    _PROFILE['fp4_gemm'][0] += 1
    _PROFILE['fp4_gemm'][1] += _time.perf_counter() - _t0
    return out.view(out_shape)


# ============================================================
# 5. sparse_attn — Vectorized sparse attention (no Python loops)
# ============================================================

def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor, softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention via index gathering.
    q: [b, s, h, d], kv: [b, n, d], attn_sink: [h], topk_idxs: [b, s, topk]
    Returns: [b, s, h, d]"""
    _t0 = _time.perf_counter()
    b, s, h, d = q.shape
    topk = topk_idxs.size(-1)

    if not hasattr(sparse_attn, '_logged'):
        print(f"[kernel_sm121] sparse_attn: VECTORIZED path, q={q.shape} kv={kv.shape} topk={topk}", flush=True)
        sparse_attn._logged = True

    # Gather KV: flatten b*s, use kv[bi] indexing per batch
    idx_clamped = topk_idxs.clamp(min=0)  # [b, s, topk]

    # For batch=1 (typical), direct index into kv[0]
    # For general case, gather per-batch
    kv_gathered = kv[:, None, :, :].expand(b, s, kv.size(1), d)  # view, no copy
    idx_exp = idx_clamped.unsqueeze(-1).expand(b, s, topk, d)
    kv_gathered = torch.gather(kv_gathered, 2, idx_exp).float()  # [b, s, topk, d]

    # Scores: [b, s, h, topk] via einsum (vectorized over all s positions)
    q_f = q.float()
    scores = torch.einsum('bshd,bstd->bsht', q_f, kv_gathered) * softmax_scale

    # Mask invalid
    valid = (topk_idxs >= 0).unsqueeze(2).expand(b, s, h, topk)
    scores = scores.masked_fill(~valid, float('-inf'))

    # Softmax with attn_sink
    sink = attn_sink.float().view(1, 1, h, 1).expand(b, s, h, 1)
    all_scores = torch.cat([scores, sink], dim=-1)
    weights = F.softmax(all_scores, dim=-1)
    attn_weights = weights[:, :, :, :topk]

    o = torch.einsum('bsht,bstd->bshd', attn_weights, kv_gathered)
    _PROFILE['sparse_attn'][0] += 1
    _PROFILE['sparse_attn'][1] += _time.perf_counter() - _t0
    return o.to(q.dtype)


# ============================================================
# 6. hc_split_sinkhorn — Hyper-Connection with Sinkhorn normalization
# ============================================================

def hc_split_sinkhorn(
    mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
    hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6,
):
    _t0 = _time.perf_counter()
    b, s, _ = mixes.shape
    hc = hc_mult
    mix_hc = (2 + hc) * hc

    flat = mixes.view(-1, mix_hc).float()
    base = hc_base.float()
    scale = hc_scale.float()
    n = flat.size(0)

    pre_raw = flat[:, :hc] * scale[0] + base[:hc]
    pre = torch.sigmoid(pre_raw) + eps

    post_raw = flat[:, hc:2*hc] * scale[1] + base[hc:2*hc]
    post = 2 * torch.sigmoid(post_raw)

    comb_raw = flat[:, 2*hc:].view(n, hc, hc) * scale[2] + base[2*hc:].view(hc, hc)
    comb = F.softmax(comb_raw, dim=-1) + eps

    col_sum = comb.sum(dim=-2, keepdim=True) + eps
    comb = comb / col_sum
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True) + eps
        comb = comb / row_sum
        col_sum = comb.sum(dim=-2, keepdim=True) + eps
        comb = comb / col_sum

    pre = pre.view(b, s, hc)
    post = post.view(b, s, hc)
    comb = comb.view(b, s, hc, hc)
    _PROFILE['hc_sinkhorn'][0] += 1
    _PROFILE['hc_sinkhorn'][1] += _time.perf_counter() - _t0
    return pre, post, comb
