"""
kernel_sm121.py — Pure PyTorch fallback kernels for DeepSeek V4 Flash on sm_121 (DGX Spark).

Drop-in replacement for the official TileLang-based kernel.py.
All 6 functions have identical signatures. Slow but correct.

Usage: in model.py, change
    from kernel import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn
to:
    from kernel_sm121 import act_quant, fp4_act_quant, fp8_gemm, fp4_gemm, sparse_attn, hc_split_sinkhorn
"""

import torch
import torch.nn.functional as F
from typing import Optional


# ============================================================
# 1. act_quant — Block-wise FP8 quantization
# ============================================================

def _round_scale_pow2(scale: torch.Tensor) -> torch.Tensor:
    """Round scale to nearest power of 2 (MXFP format)."""
    return torch.exp2(torch.ceil(torch.log2(scale)))


def act_quant(
    x: torch.Tensor, block_size: int = 128, scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32, inplace: bool = False,
) -> torch.Tensor:
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
        return x
    else:
        y = clamped.view(M, N).to(torch.float8_e4m3fn).view(*orig_shape[:-1], N)
        s = scale.to(scale_dtype).view(*orig_shape[:-1], n_blocks)
        return y, s


# ============================================================
# 2. fp4_act_quant — Block-wise FP4 simulation quantization
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
        # Simulate FP4 precision: round to FP4 representable values then dequant
        # FP4 e2m1 representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
        fp4_values = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device, dtype=torch.float32
        )
        signs = clamped.sign()
        abs_vals = clamped.abs()
        # Find nearest FP4 value by broadcasting
        diffs = (abs_vals.unsqueeze(-1) - fp4_values).abs()
        nearest_idx = diffs.argmin(dim=-1)
        quantized = signs * fp4_values[nearest_idx]
        result = (quantized * scale.unsqueeze(-1)).view(orig_shape).to(x.dtype)
        x.copy_(result)
        return x
    else:
        raise NotImplementedError("fp4_act_quant non-inplace not needed for sm_121 fallback")


# ============================================================
# 3. fp8_gemm — FP8 x FP8 GEMM with per-block scaling
# ============================================================

def fp8_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T with per-128 block FP8 scaling."""
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    out_shape = (*a.shape[:-1], N)

    a_flat = a.view(M, K).to(torch.bfloat16)
    b_flat = b.to(torch.bfloat16)

    a_s_flat = a_s.view(M, -1).float()
    b_s_flat = b_s.float()

    block_size = 128
    n_blocks_k = K // block_size

    # Dequantize A: multiply each block by its scale
    a_deq = a_flat.float().view(M, n_blocks_k, block_size) * a_s_flat.unsqueeze(-1)
    a_deq = a_deq.view(M, K).to(torch.bfloat16)

    # Dequantize B: b_s is [ceil(N/128), ceil(K/128)]
    n_blocks_n = b_s_flat.size(0)
    b_deq = b_flat.float().view(N, n_blocks_k, block_size)
    # b_s[i, j] applies to rows [i*128:(i+1)*128] and cols [j*128:(j+1)*128]
    # but b_s shape is [ceil(N/128), ceil(K/128)], expand to per-element
    b_scale_expanded = b_s_flat.unsqueeze(-1).expand(n_blocks_n, n_blocks_k, block_size)
    b_scale_full = b_scale_expanded.reshape(n_blocks_n, K)
    # Each row group of 128 in B gets the same K-scale pattern
    b_row_scales = b_scale_full.unsqueeze(1).expand(n_blocks_n, min(128, N), K)
    # Handle last block which may be smaller
    actual_rows = []
    for i in range(n_blocks_n):
        start = i * 128
        end = min(start + 128, N)
        actual_rows.append(end - start)
    b_deq_parts = []
    for i in range(n_blocks_n):
        start = i * 128
        end = min(start + 128, N)
        rows = b_flat[start:end].float().view(end - start, n_blocks_k, block_size)
        row_scale = b_s_flat[i].unsqueeze(-1).expand(n_blocks_k, block_size)
        rows = rows * row_scale.unsqueeze(0)
        b_deq_parts.append(rows.view(end - start, K))
    b_deq = torch.cat(b_deq_parts, dim=0).to(torch.bfloat16)

    c = torch.mm(a_deq, b_deq.t())
    return c.view(out_shape)


# ============================================================
# 4. fp4_gemm — FP8 act x FP4 weight GEMM with per-block scaling
# ============================================================

FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
], dtype=torch.float32)


def _dequant_fp4(b: torch.Tensor) -> torch.Tensor:
    """Dequantize FP4 (float4_e2m1fn_x2) to float32 via int8 view + lookup table.
    b: [N, K//2] in float4_e2m1fn_x2 (two FP4 values packed per byte).
    Returns: [N, K] in float32."""
    raw = b.view(torch.uint8)  # reinterpret as uint8
    low = raw & 0x0F
    high = (raw >> 4) & 0x0F
    table = FP4_TABLE.to(raw.device)
    return torch.stack([table[low.long()], table[high.long()]], dim=-1).flatten(-2)


def fp4_gemm(
    a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T.
    A: FP8 with per-128 scale. B: FP4 (float4_e2m1fn_x2) with per-32 E8M0 scale."""
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    out_shape = (*a.shape[:-1], N)

    # Dequantize A (FP8 with per-128 scale)
    a_flat = a.view(M, K).float()
    a_s_flat = a_s.view(M, -1).float()
    act_block = 128
    n_act_blocks = K // act_block
    a_deq = a_flat.view(M, n_act_blocks, act_block) * a_s_flat.unsqueeze(-1)
    a_deq = a_deq.view(M, K).to(torch.bfloat16)

    # Dequantize B (FP4 via manual unpack + per-32 E8M0 scale)
    b_fp32 = _dequant_fp4(b)  # [N, K]
    b_s_flat = b_s.float()    # E8M0 → float32, shape [N, K//32]
    weight_block = 32
    n_weight_blocks = K // weight_block
    b_deq = b_fp32.view(N, n_weight_blocks, weight_block) * b_s_flat.unsqueeze(-1)
    b_deq = b_deq.view(N, K).to(torch.bfloat16)

    c = torch.mm(a_deq, b_deq.t())
    return c.view(out_shape)


# ============================================================
# 5. sparse_attn — Sparse attention with index gathering + online softmax
# ============================================================

def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor, softmax_scale: float,
) -> torch.Tensor:
    """Sparse multi-head attention via index gathering.
    q: [b, s, h, d], kv: [b, n, d], attn_sink: [h], topk_idxs: [b, s, topk]
    Returns: [b, s, h, d]"""
    b, s, h, d = q.shape
    topk = topk_idxs.size(-1)

    q_f = q.float()
    kv_f = kv.float()

    o = torch.zeros(b, s, h, d, device=q.device, dtype=torch.float32)

    for bi in range(b):
        for si in range(s):
            idxs = topk_idxs[bi, si]  # [topk]
            valid = idxs >= 0
            kv_gathered = kv_f[bi, idxs.clamp(min=0)]  # [topk, d]

            q_vec = q_f[bi, si]  # [h, d]
            scores = torch.mm(q_vec, kv_gathered.t()) * softmax_scale  # [h, topk]
            scores[:, ~valid] = float('-inf')

            # Include attn_sink as an extra "score" for the softmax denominator
            sink_scores = attn_sink.float().unsqueeze(-1)  # [h, 1]
            all_scores = torch.cat([scores, sink_scores], dim=-1)  # [h, topk+1]
            weights = F.softmax(all_scores, dim=-1)  # [h, topk+1]

            attn_weights = weights[:, :topk]  # [h, topk] — drop sink weight
            o[bi, si] = torch.mm(attn_weights, kv_gathered)  # [h, d]

    return o.to(q.dtype)


# ============================================================
# 6. hc_split_sinkhorn — Hyper-Connection with Sinkhorn normalization
# ============================================================

def hc_split_sinkhorn(
    mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor,
    hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6,
):
    """Split mixes into pre, post, comb matrices with Sinkhorn normalization.
    mixes: [b, s, (2+hc)*hc], hc_scale: [3], hc_base: [(2+hc)*hc]
    Returns: pre [b,s,hc], post [b,s,hc], comb [b,s,hc,hc]"""
    b, s, _ = mixes.shape
    hc = hc_mult
    mix_hc = (2 + hc) * hc

    flat = mixes.view(-1, mix_hc).float()
    base = hc_base.float()
    scale = hc_scale.float()
    n = flat.size(0)

    pre_raw = flat[:, :hc] * scale[0] + base[:hc]
    pre = torch.sigmoid(pre_raw) + eps  # [n, hc]

    post_raw = flat[:, hc:2*hc] * scale[1] + base[hc:2*hc]
    post = 2 * torch.sigmoid(post_raw)  # [n, hc]

    comb_raw = flat[:, 2*hc:].view(n, hc, hc) * scale[2] + base[2*hc:].view(hc, hc)

    # softmax along last dim + eps
    comb = F.softmax(comb_raw, dim=-1) + eps  # [n, hc, hc]

    # First column normalization
    col_sum = comb.sum(dim=-2, keepdim=True) + eps  # [n, 1, hc]
    comb = comb / col_sum

    # Sinkhorn iterations
    for _ in range(sinkhorn_iters - 1):
        row_sum = comb.sum(dim=-1, keepdim=True) + eps  # [n, hc, 1]
        comb = comb / row_sum
        col_sum = comb.sum(dim=-2, keepdim=True) + eps  # [n, 1, hc]
        comb = comb / col_sum

    pre = pre.view(b, s, hc)
    post = post.view(b, s, hc)
    comb = comb.view(b, s, hc, hc)
    return pre, post, comb
