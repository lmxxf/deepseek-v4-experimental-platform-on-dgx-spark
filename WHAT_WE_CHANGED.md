# What We Changed (and Why)

[中文版](WHAT_WE_CHANGED_zh.md)

This project runs DeepSeek V4 Flash (280B) on DGX Spark **without modifying any official DeepSeek code**. Instead, we provide drop-in replacements that intercept at the import level.

---

## Architecture Overview

```
DeepSeek Official Code (untouched)        Our Replacements
─────────────────────────────────────      ─────────────────────────────────
inference/model.py                         
  └── from kernel import ...         ──→   kernel_sm121.py  (6 kernel fallbacks)
  └── from fast_hadamard_transform   ──→   fast_hadamard_transform.py
  └── load_model(file)               ──→   weight_loader.py (streaming loader)
```

`test_dual_node.py` does two things before importing `model.py`:
```python
sys.modules['kernel'] = kernel_sm121           # redirect kernel imports
sys.path.insert(0, '/workspace')               # fast_hadamard_transform.py found here
```

That's it. `model.py` runs unmodified — it just finds our implementations when it does `from kernel import ...`.

---

## 1. kernel_sm121.py — Replacing TileLang Kernels

**Problem**: DeepSeek's `kernel.py` uses [TileLang](https://github.com/tile-ai/tilelang) JIT-compiled GPU kernels. TileLang 0.1.8 only supports sm_75 (Turing), sm_90 (Hopper), and sm_100 (datacenter Blackwell). DGX Spark's GB10 is sm_121 — not supported.

**Root cause**: sm_120 is the consumer Blackwell family (RTX 5090, etc.). Nobody runs distributed inference on consumer GPUs, so nobody builds kernel libraries for them.

**Solution**: Pure PyTorch implementations of all 6 kernels, with identical function signatures.

### `act_quant(x, block_size, scale_fmt, scale_dtype, inplace)`

Block-wise FP8 quantization. For each 128-element block:
1. Compute `absmax` of the block
2. Derive scale: `scale = absmax / 448.0` (FP8 e4m3 max)
3. Optional: round scale to power-of-2 for MXFP format
4. Clamp and cast to `float8_e4m3fn`

Inplace mode: quantize then immediately dequantize back to BF16 (simulates quantization noise).

### `fp4_act_quant(x, block_size, inplace)`

Block-wise FP4 simulation quantization. Same pattern as `act_quant`, but:
- FP4 max = 6.0, block size = 32
- Uses `FP4_TABLE` lookup to find nearest representable FP4 value
- Only inplace mode needed (model uses it to simulate FP4 precision loss)

### `fp8_gemm(a, a_s, b, b_s, scale_dtype)`

FP8×FP8 matrix multiplication with per-block scaling:
1. Dequantize A: `A_deq[i,k] = A_fp8[i,k] * a_scale[i, k//128]`
2. Dequantize B: `B_deq[n,k] = B_fp8[n,k] * b_scale[n//128, k//128]`
3. `C = A_deq @ B_deq.T` in BF16

Note: `torch._scaled_mm` works on sm_121 for FP8 GEMM, but doesn't support per-block scaling (only per-tensor). So we dequantize manually.

### `fp4_gemm(a, a_s, b, b_s, scale_dtype)`

FP8 activation × FP4 weight GEMM:
1. Dequantize A same as `fp8_gemm`
2. Dequantize B: **cannot use `float4_e2m1fn_x2.to(float32)`** — this operation is unimplemented in PyTorch on both CPU and GPU. Instead:
   - View FP4 tensor as `uint8`
   - Extract low/high nibbles via bit operations
   - Look up values in `FP4_TABLE = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]`
   - Multiply by per-32-element E8M0 scale
3. `C = A_deq @ B_deq.T` in BF16

### `sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)`

Sparse multi-head attention with index gathering:
- `q`: [batch, seq, heads, head_dim]
- `kv`: [batch, n_positions, head_dim]
- Gathers top-k KV positions by index, computes Q·K^T, applies softmax with learnable `attn_sink` bias

Current implementation uses Python for-loops (slow). Sufficient for research — activation extraction only needs one forward pass.

### `hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)`

Hyper-Connection Sinkhorn normalization:
1. Split `mixes` into `pre` (sigmoid), `post` (2×sigmoid), `comb` (matrix)
2. Apply softmax + epsilon to `comb`
3. Run 20 iterations of Sinkhorn normalization (alternating row/column normalization)

Returns pre weights, post weights, and combination matrix for the 4-stream residual.

---

## 2. fast_hadamard_transform.py — Replacing CUDA Kernel

**Problem**: `fast_hadamard_transform` is a pip package with CUDA kernels that won't compile on sm_121.

**Solution**: 32-line PyTorch implementation. Constructs the Hadamard matrix recursively, caches it, multiplies with input. Negligible overhead — the matrix is tiny (128×128).

---

## 3. weight_loader.py — Streaming Weight Loading

**Problem**: DeepSeek's `generate.py` uses `safetensors.torch.load_model()` which loads the entire weight file into CPU memory first. On DGX Spark's unified memory (128GB shared between CPU and GPU), this causes OOM:

```
Model skeleton on GPU:    ~82 GB
Weight file on CPU (mmap): ~77 GB
Total:                    ~159 GB > 128 GB → OOM
```

We tried several approaches:
- `safe_open(device="cuda")`: still mmaps the file, active pages compete with GPU allocations
- `meta` device skeleton + streaming: works but causes 6 rounds of meta tensor debugging hell
- Batched loading with periodic file close: mmap pages still accumulate

**Solution**: Skip `convert.py` entirely. Load directly from HuggingFace's original 46 shards (~3.5GB each):

```python
for shard_file in sorted(glob("*.safetensors")):
    with safe_open(shard_file, framework="pt", device="cuda") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # rename key, TP shard, assign to model
            del tensor
    # shard closed → mmap pages released
```

Each shard is opened, processed, and closed independently. Peak mmap = 3.5GB. This is the same approach vLLM uses internally.

### Key mapping

HuggingFace uses different parameter names than DeepSeek's official code:

| HuggingFace | Official | TP dimension |
|---|---|---|
| `model.layers.X.self_attn.q_proj` | `layers.X.attn.wq` | dim 0 |
| `model.layers.X.self_attn.o_proj` | `layers.X.attn.wo` | dim 1 |
| `model.layers.X.mlp.gate_proj` | `layers.X.ffn.w1` | dim 0 |
| `model.layers.X.mlp.down_proj` | `layers.X.ffn.w2` | dim 1 |
| `embed_tokens` | `embed` | dim 0 |
| `lm_head` | `head` | dim 0 |
| ... | ... | ... |

### TP sharding

Weights are sharded on-the-fly based on `rank` and `world_size`:
- Non-expert weights: `tensor.narrow(dim, rank * shard_size, shard_size)`
- Expert weights: skip experts not belonging to this rank (`expert_id // n_local_experts != rank`)

### Special handling

- **`wo_a` weights**: scale is fused into weight and converted to BF16 (same as `convert.py`)
- **FP4 expert weights**: `int8` viewed as `float4_e2m1fn_x2` (zero-copy reinterpret)
- **`weight.scale` linkage**: after loading, re-links `module.weight.scale = module.scale` for all `Linear` layers

---

## 4. test_dual_node.py — Monkey Patches

Two small patches applied before model runs:

### dtype mismatch fix

Some code paths in `model.py` call `.float()` on activations before passing to BF16 weights. On datacenter GPUs with TileLang, mixed-dtype `F.linear` works. On our pure PyTorch path, it doesn't.

```python
def _patched_linear(x, weight, bias=None):
    if weight.dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return _original_linear(x, weight, bias)
    return F.linear(x.to(weight.dtype), weight, bias)
```

### `ParallelHead.get_logits` fix

Same issue: `x[:, -1].float()` with BF16 weight.

```python
def _patched_get_logits(self, x):
    return F.linear(x[:, -1].float(), self.weight.float())
```

---

## What We Did NOT Change

- **`model.py`**: runs completely unmodified
- **`generate.py`**: not used (we have our own entry point), but compatible
- **`encoding_dsv4.py`**: used as-is for chat template formatting
- **Model weights**: loaded from original HuggingFace format, no conversion needed
- **NCCL / PyTorch Distributed**: standard `dist.init_process_group("nccl")`, no custom communication code
