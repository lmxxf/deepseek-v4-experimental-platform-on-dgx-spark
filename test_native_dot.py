"""Test native FP8 dot and FP4 dot_scaled on sm_121 in vllm-node-sm120 container.

Three tests:
1. tl.dot(fp8, fp8) — skip .to(float32), let Triton use FP8 tensor core
2. tl.dot_scaled("e4m3","e2m1") — FP8 act × FP4 weight, native block_scale MMA
3. Speed comparison: current (dequant→f32→dot) vs native

Run: docker run --rm --gpus all -v $PWD:/workspace vllm-node-sm120 python3 /workspace/test_native_dot.py
"""

import time
import torch
import triton
import triton.language as tl

print(f"PyTorch {torch.__version__}, Triton {triton.__version__}")
print(f"GPU: {torch.cuda.get_device_name()}, SM {torch.cuda.get_device_capability()}")


# ════════════════════════════════════════════════════════════════
# Test 1: tl.dot(fp8, fp8) — native FP8 tensor core
# ════════════════════════════════════════════════════════════════

@triton.jit
def _fp8_dot_native_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K: tl.constexpr,
    a_stride_m, a_stride_k,
    b_stride_n, b_stride_k,
    out_stride_m, out_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """FP8 × FP8 → f32 accumulator, no dequant."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
        )
        b = tl.load(
            b_ptr + offs_n[:, None] * b_stride_n + offs_k[None, :] * b_stride_k,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0,
        )
        # Key: NO .to(tl.float32) — let Triton dispatch to FP8 mma
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _fp8_dot_dequant_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K: tl.constexpr,
    a_stride_m, a_stride_k,
    b_stride_n, b_stride_k,
    out_stride_m, out_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Current approach: FP8 → float32 → dot(f32, f32)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * a_stride_m + offs_k[None, :] * a_stride_k,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0,
        ).to(tl.float32)  # ← current: cast to f32
        b = tl.load(
            b_ptr + offs_n[:, None] * b_stride_n + offs_k[None, :] * b_stride_k,
            mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0,
        ).to(tl.float32)  # ← current: cast to f32
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)

    tl.store(
        out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def test_fp8_dot():
    print("\n" + "="*60)
    print("Test 1: tl.dot(fp8, fp8) — native FP8 tensor core")
    print("="*60)

    M, K, N = 64, 7168, 4096

    a_bf16 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
    b_bf16 = torch.randn(N, K, device='cuda', dtype=torch.bfloat16)

    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.to(torch.float8_e4m3fn)

    ref = (a_fp8.float() @ b_fp8.float().t()).bfloat16()

    BLOCK_M, BLOCK_N, BLOCK_K = 16, 32, 128
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    # Native
    out_native = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    try:
        _fp8_dot_native_kernel[grid](
            a_fp8, b_fp8, out_native,
            M, N, K,
            a_fp8.stride(0), a_fp8.stride(1),
            b_fp8.stride(0), b_fp8.stride(1),
            out_native.stride(0), out_native.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        torch.cuda.synchronize()
        diff = (out_native.float() - ref.float()).abs().max().item()
        print(f"  Native FP8 dot: compiled ✅, max_diff={diff:.4f}")
    except Exception as e:
        print(f"  Native FP8 dot: FAILED ❌ — {type(e).__name__}: {e}")
        return

    # Dequant (current)
    out_deq = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
    _fp8_dot_dequant_kernel[grid](
        a_fp8, b_fp8, out_deq,
        M, N, K,
        a_fp8.stride(0), a_fp8.stride(1),
        b_fp8.stride(0), b_fp8.stride(1),
        out_deq.stride(0), out_deq.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    torch.cuda.synchronize()

    # Bench
    def bench(fn, n_warmup=5, n_iter=50):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000

    def run_native():
        _fp8_dot_native_kernel[grid](
            a_fp8, b_fp8, out_native, M, N, K,
            a_fp8.stride(0), a_fp8.stride(1),
            b_fp8.stride(0), b_fp8.stride(1),
            out_native.stride(0), out_native.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

    def run_dequant():
        _fp8_dot_dequant_kernel[grid](
            a_fp8, b_fp8, out_deq, M, N, K,
            a_fp8.stride(0), a_fp8.stride(1),
            b_fp8.stride(0), b_fp8.stride(1),
            out_deq.stride(0), out_deq.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

    for test_M, label in [(1, "M=1 (gen)"), (12, "M=12 (short)"), (64, "M=64"), (384, "M=384"), (1024, "M=1024 (long)")]:
        a_fp8_t = torch.randn(test_M, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        out_n = torch.empty(test_M, N, device='cuda', dtype=torch.bfloat16)
        out_d = torch.empty(test_M, N, device='cuda', dtype=torch.bfloat16)
        grid_t = (triton.cdiv(test_M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        def run_n():
            _fp8_dot_native_kernel[grid_t](
                a_fp8_t, b_fp8, out_n, test_M, N, K,
                a_fp8_t.stride(0), a_fp8_t.stride(1),
                b_fp8.stride(0), b_fp8.stride(1),
                out_n.stride(0), out_n.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )

        def run_d():
            _fp8_dot_dequant_kernel[grid_t](
                a_fp8_t, b_fp8, out_d, test_M, N, K,
                a_fp8_t.stride(0), a_fp8_t.stride(1),
                b_fp8.stride(0), b_fp8.stride(1),
                out_d.stride(0), out_d.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            )

        t_n = bench(run_n)
        t_d = bench(run_d)
        speedup = t_d / t_n if t_n > 0 else 0
        print(f"  {label:20s}: native={t_n:.3f}ms  dequant={t_d:.3f}ms  speedup={speedup:.2f}x")


# ════════════════════════════════════════════════════════════════
# Test 2: tl.dot_scaled("e4m3","e2m1") — FP8 act × FP4 weight
# ════════════════════════════════════════════════════════════════

@triton.jit
def _dot_scaled_fp8xfp4_kernel(
    a_ptr, a_stride_m, a_stride_k,
    a_scale_ptr, as_stride_m, as_stride_k,
    b_ptr, b_stride_k, b_stride_n,
    b_scale_ptr, bs_stride_n, bs_stride_k,
    d_ptr, d_stride_m, d_stride_n,
    M, N, K: tl.constexpr,
    K_PACKED: tl.constexpr,
    K_SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """D[M,N] = dot_scaled(A_fp8[M,K_packed_fp8], B_fp4[K_packed_fp4, N])
    A: [M, K] uint8 (FP8 e4m3), a_scale: [M, K//128] uint8 (E8M0)
    B: [K//2, N] uint8 (packed FP4 e2m1), b_scale: [N, K//32] uint8 (E8M0)

    dot_scaled chunk = 32 FP8 bytes (A) × 16 FP4 packed bytes (B) = 32 K values
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    CHUNK_K: tl.constexpr = 32
    CHUNK_K_PACKED: tl.constexpr = 16

    for k0 in range(0, K, CHUNK_K):
        k_offs = k0 + tl.arange(0, CHUNK_K)
        kp_start = k0 // 2
        kp_offs = kp_start + tl.arange(0, CHUNK_K_PACKED)

        a_chunk = tl.load(
            a_ptr + m_offs[:, None] * a_stride_m + k_offs[None, :] * a_stride_k,
            mask=(m_offs[:, None] < M) & (k_offs[None, :] < K), other=0,
        )
        b_chunk = tl.load(
            b_ptr + kp_offs[:, None] * b_stride_k + n_offs[None, :] * b_stride_n,
            mask=(kp_offs[:, None] < K_PACKED) & (n_offs[None, :] < N), other=0,
        )

        scale_idx = k0 // 32
        a_sc = tl.load(
            a_scale_ptr + m_offs[:, None] * as_stride_m + scale_idx,
            mask=m_offs[:, None] < M, other=127,
        )
        b_sc = tl.load(
            b_scale_ptr + n_offs[:, None] * bs_stride_n + scale_idx,
            mask=n_offs[:, None] < N, other=127,
        )

        acc += tl.dot_scaled(a_chunk, a_sc, "e4m3", b_chunk, b_sc, "e2m1")

    tl.store(
        d_ptr + m_offs[:, None] * d_stride_m + n_offs[None, :] * d_stride_n,
        acc.to(tl.bfloat16),
        mask=(m_offs[:, None] < M) & (n_offs[None, :] < N),
    )


@triton.jit
def _fp4_dequant_dot_kernel(
    a_ptr, a_scale_ptr, b_ptr, b_scale_ptr, out_ptr,
    M, N, K: tl.constexpr,
    a_stride_m, a_stride_k,
    as_stride_m, as_stride_kb,
    b_stride_n, b_stride_kh,
    bs_stride_n, bs_stride_kb,
    out_stride_m, out_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Current approach: manual FP4 decode → f32 → split even/odd → two tl.dot(f32,f32)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_kh = tl.arange(0, BLOCK_K // 2)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        a_even_k = k0 + offs_kh * 2
        a_even = tl.load(
            a_ptr + offs_m[:, None] * a_stride_m + a_even_k[None, :] * a_stride_k,
            mask=(offs_m[:, None] < M) & (a_even_k[None, :] < K), other=0.0,
        ).to(tl.float32)
        a_odd_k = a_even_k + 1
        a_odd = tl.load(
            a_ptr + offs_m[:, None] * a_stride_m + a_odd_k[None, :] * a_stride_k,
            mask=(offs_m[:, None] < M) & (a_odd_k[None, :] < K), other=0.0,
        ).to(tl.float32)

        act_kb = k0 // 128
        a_s = tl.load(
            a_scale_ptr + offs_m * as_stride_m + act_kb * as_stride_kb,
            mask=offs_m < M, other=0.0,
        ).to(tl.float32)

        kh = k0 // 2 + offs_kh
        b_packed = tl.load(
            b_ptr + offs_n[:, None] * b_stride_n + kh[None, :] * b_stride_kh,
            mask=(offs_n[:, None] < N) & (kh[None, :] < K // 2), other=0,
        )

        # Manual FP4 decode — inline (Triton JIT doesn't support nested def)
        low_nibble = b_packed & 0x0F
        low_sign = (low_nibble >> 3) & 1
        low_mag = low_nibble & 0x07
        low_e = low_mag >> 1
        low_m = low_mag & 1
        low_norm = (2 + low_m).to(tl.float32) * tl.exp2((low_e - 1).to(tl.float32)) * 0.5
        low_sub = low_m.to(tl.float32) * 0.5
        low_val = tl.where(low_e == 0, low_sub, low_norm)
        b_low = tl.where(low_sign == 1, -low_val, low_val)

        high_nibble = (b_packed >> 4) & 0x0F
        high_sign = (high_nibble >> 3) & 1
        high_mag = high_nibble & 0x07
        high_e = high_mag >> 1
        high_m = high_mag & 1
        high_norm = (2 + high_m).to(tl.float32) * tl.exp2((high_e - 1).to(tl.float32)) * 0.5
        high_sub = high_m.to(tl.float32) * 0.5
        high_val = tl.where(high_e == 0, high_sub, high_norm)
        b_high = tl.where(high_sign == 1, -high_val, high_val)

        even_wkb = a_even_k // 32
        odd_wkb = a_odd_k // 32
        bs_even = tl.load(
            b_scale_ptr + offs_n[:, None] * bs_stride_n + even_wkb[None, :] * bs_stride_kb,
            mask=(offs_n[:, None] < N) & (even_wkb[None, :] < K // 32), other=1.0,
        ).to(tl.float32)
        bs_odd = tl.load(
            b_scale_ptr + offs_n[:, None] * bs_stride_n + odd_wkb[None, :] * bs_stride_kb,
            mask=(offs_n[:, None] < N) & (odd_wkb[None, :] < K // 32), other=1.0,
        ).to(tl.float32)

        acc += tl.dot(a_even, tl.trans(b_low * bs_even), out_dtype=tl.float32) * a_s[:, None]
        acc += tl.dot(a_odd, tl.trans(b_high * bs_odd), out_dtype=tl.float32) * a_s[:, None]

    tl.store(
        out_ptr + offs_m[:, None] * out_stride_m + offs_n[None, :] * out_stride_n,
        acc.to(tl.bfloat16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def test_dot_scaled():
    print("\n" + "="*60)
    print("Test 2: tl.dot_scaled('e4m3','e2m1') — FP8×FP4 native")
    print("="*60)

    M, K, N = 64, 7168, 4096
    K_PACKED = K // 2
    K_SCALE_ACT = K // 128
    K_SCALE_W = K // 32

    a_fp8 = torch.randn(M, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    a_fp8_u8 = a_fp8.view(torch.uint8)
    a_scale = torch.full((M, K_SCALE_ACT), 127, device='cuda', dtype=torch.uint8)

    b_packed_nk = torch.randint(0, 256, (N, K_PACKED), device='cuda', dtype=torch.uint8)
    b_packed_kn = b_packed_nk.t().contiguous()
    b_scale = torch.full((N, K_SCALE_W), 127, device='cuda', dtype=torch.uint8)

    # For dot_scaled, a_scale needs K//32 not K//128 (one scale per 32 values)
    a_scale_32 = torch.full((M, K // 32), 127, device='cuda', dtype=torch.uint8)

    d = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)

    BLOCK_M, BLOCK_N = 16, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    try:
        _dot_scaled_fp8xfp4_kernel[grid](
            a_fp8_u8, a_fp8_u8.stride(0), a_fp8_u8.stride(1),
            a_scale_32, a_scale_32.stride(0), a_scale_32.stride(1),
            b_packed_kn, b_packed_kn.stride(0), b_packed_kn.stride(1),
            b_scale, b_scale.stride(0), b_scale.stride(1),
            d, d.stride(0), d.stride(1),
            M, N, K, K_PACKED, K_SCALE_W,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        torch.cuda.synchronize()
        print(f"  dot_scaled e4m3×e2m1: compiled ✅")
        print(f"  d[:2,:4] = {d[:2,:4]}")
    except Exception as e:
        print(f"  dot_scaled e4m3×e2m1: FAILED ❌ — {type(e).__name__}: {e}")
        return

    # Bench: dot_scaled vs current dequant approach
    # Need E8M0→f32 converted scales for dequant kernel
    a_scale_f32 = torch.pow(2.0, a_scale.float() - 127.0).contiguous()
    b_scale_f32 = torch.pow(2.0, b_scale.float() - 127.0).contiguous()

    def bench(fn, n_warmup=5, n_iter=50):
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_iter * 1000

    for test_M, label in [(1, "M=1 (gen)"), (12, "M=12 (short)"), (64, "M=64"), (384, "M=384"), (1024, "M=1024 (long)")]:
        a_t = torch.randn(test_M, K, device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        a_t_u8 = a_t.view(torch.uint8)
        a_sc_32 = torch.full((test_M, K // 32), 127, device='cuda', dtype=torch.uint8)
        a_sc_128 = torch.full((test_M, K // 128), 127, device='cuda', dtype=torch.uint8)
        a_sc_f32 = torch.ones(test_M, K // 128, device='cuda', dtype=torch.float32)
        d_n = torch.empty(test_M, N, device='cuda', dtype=torch.bfloat16)
        d_d = torch.empty(test_M, N, device='cuda', dtype=torch.bfloat16)
        grid_t = (triton.cdiv(test_M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        def run_native():
            _dot_scaled_fp8xfp4_kernel[grid_t](
                a_t_u8, a_t_u8.stride(0), a_t_u8.stride(1),
                a_sc_32, a_sc_32.stride(0), a_sc_32.stride(1),
                b_packed_kn, b_packed_kn.stride(0), b_packed_kn.stride(1),
                b_scale, b_scale.stride(0), b_scale.stride(1),
                d_n, d_n.stride(0), d_n.stride(1),
                test_M, N, K, K_PACKED, K_SCALE_W,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            )

        def run_dequant():
            _fp4_dequant_dot_kernel[grid_t](
                a_t, a_sc_f32, b_packed_nk, b_scale_f32, d_d,
                test_M, N, K,
                a_t.stride(0), a_t.stride(1),
                a_sc_f32.stride(0), a_sc_f32.stride(1),
                b_packed_nk.stride(0), b_packed_nk.stride(1),
                b_scale_f32.stride(0), b_scale_f32.stride(1),
                d_d.stride(0), d_d.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=128,
            )

        t_n = bench(run_native)
        t_d = bench(run_dequant)
        speedup = t_d / t_n if t_n > 0 else 0
        print(f"  {label:20s}: dot_scaled={t_n:.3f}ms  dequant={t_d:.3f}ms  speedup={speedup:.2f}x")


# ════════════════════════════════════════════════════════════════
# Test 3: hc_sinkhorn iteration count
# ════════════════════════════════════════════════════════════════

def test_sinkhorn_iters():
    print("\n" + "="*60)
    print("Test 3: hc_sinkhorn convergence — 20 iters vs 5 iters")
    print("="*60)

    def sinkhorn(comb, n_iters, eps=1e-6):
        for _ in range(n_iters):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return comb

    torch.manual_seed(42)
    n = 10000
    comb_raw = torch.randn(n, 4, 4, device='cuda')
    comb_init = torch.softmax(comb_raw, dim=-1) + 1e-6

    ref = sinkhorn(comb_init.clone(), 20)

    for n_iter in [3, 5, 7, 10]:
        result = sinkhorn(comb_init.clone(), n_iter)
        diff = (result - ref).abs().max().item()
        row_err = (result.sum(-1) - 1.0).abs().max().item()
        col_err = (result.sum(-2) - 1.0).abs().max().item()
        print(f"  {n_iter:2d} iters: max_diff_vs_20={diff:.2e}, row_err={row_err:.2e}, col_err={col_err:.2e}")


if __name__ == "__main__":
    test_fp8_dot()
    test_dot_scaled()
    test_sinkhorn_iters()
    print("\nDone.")
