# DeepSeek V4 Flash Experimental Platform on DGX Spark

**Extract intermediate activations from DeepSeek V4 Flash (280B) on two DGX Spark nodes.**

[中文文档](README_zh.md)

Triton + PyTorch implementation — no TileLang, no DeepGEMM, no vLLM. Direct access to every layer's 4-stream Hyper-Connection residual, attention outputs, and MoE routing decisions.

## Why

vLLM is a black-box inference server. You can't hook into the forward pass to extract hidden states, attention patterns, or MoE routing decisions. This platform gives you full control over the model's internals for research purposes.

## The SM_120 Problem

DGX Spark uses GB10 GPUs (sm_121), a consumer-grade Blackwell variant. **No major ML framework supports sm_120-series for distributed inference.** This isn't because sm_121 is exotic — it's because sm_120 is consumer hardware (RTX 5090, etc.), and nobody runs multi-GPU distributed inference on consumer GPUs. There's no market, so nobody built the infrastructure.

DGX Spark is the first product to use a consumer-grade GPU architecture for datacenter-class workloads (dual-node, 256GB unified memory, 280B model). It hits an ecosystem gap that doesn't exist for datacenter Blackwell (sm_100) or consumer single-GPU use cases.

We replaced every GPU kernel with Triton + PyTorch alternatives:

| Original (TileLang) | Our Replacement | Notes |
|---|---|---|
| `act_quant` | PyTorch absmax→scale→clamp→cast | Block-wise FP8 quantization |
| `fp4_act_quant` | PyTorch FP4 simulation via lookup table | Block-wise FP4 quantization |
| `fp8_gemm` | Triton `tl.dot` with per-block scaling | FP8×FP8 GEMM |
| `fp4_gemm` | Triton `tl.dot_scaled("e4m3","e2m1")` | FP8×FP4 GEMM, 6-7x faster than manual dequant |
| `sparse_attn` | PyTorch `torch.einsum` + gather (vectorized) | Sparse attention, no Python loops |
| `hc_split_sinkhorn` | PyTorch sigmoid + softmax + Sinkhorn (5 iters) | Hyper-Connection mixing |
| `fast_hadamard_transform` | PyTorch Hadamard matrix multiply | CUDA kernel won't compile on sm_121 |

## The Unified Memory Problem

DGX Spark shares 128GB between CPU and GPU — they are the same physical memory pool. This means:

- Loading a 77GB weight file via mmap causes OOM: mmap pages + GPU allocations compete for the same 128GB
- You cannot create the model skeleton on GPU then load weights via CPU (double-buffering)
- `safe_open(device="cuda")` still uses CPU as intermediary under the hood

Our solution: stream weights directly from 46 small HuggingFace shards (~3.5GB each), opening and closing each shard independently. Peak mmap overhead is 3.5GB instead of 77GB. This is the same approach vLLM uses internally.

## Results

```
Prefill:    9.0s for 971 tokens (108 tok/s)
Generation: 153 tokens in 37.2s (4.11 tok/s)

Activation extraction (layer 28, dream SVG prompt):
  Shape: [1, 971, 4, 4096]  (batch, seq_len, 4 HC streams, hidden_dim)
  Stream norms: [108.00, 106.50, 57.75, 30.38]  ← highly asymmetric
```

Numerically correct — output matches vLLM inference server. Good enough for research: you only need one forward pass to extract activations.

## Hardware Requirements

- 2× NVIDIA DGX Spark (128GB unified memory each)
- ConnectX-7 200Gbps direct connection (QSFP56 DAC cable)
- DeepSeek V4 Flash weights (~158GB, from HuggingFace)

## Prerequisites

1. Two DGX Spark nodes networked via CX7 with SSH passwordless access
2. Docker with `vllm-node-sm120` image (from [deepseek-v4-deployment-on-dgx-spark](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark)), or any Docker image with:
   - PyTorch 2.11+ with CUDA 13.x and sm_120 support
   - Custom NCCL built for DGX Spark CX7 (from [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker))
   - `transformers`, `safetensors`
3. DeepSeek V4 Flash weights on both nodes:
   ```bash
   huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir /path/to/deepseek-v4-flash
   rsync -avP /path/to/deepseek-v4-flash user@worker-ip:/path/to/deepseek-v4-flash
   ```

## Quick Start

### 1. Clone

```bash
git clone https://github.com/lmxxf/deepseek-v4-experimental-platform-on-dgx-spark.git
cd deepseek-v4-experimental-platform-on-dgx-spark
```

### 2. Configure `run_dual_node.sh`

```bash
HEAD_IP="169.254.248.35"    # host CX7 IP
WORKER_IP="169.254.30.81"   # slave CX7 IP
WORKER_SSH="lmxxf@169.254.30.81"
MODEL_DIR="/path/to/deepseek-v4-flash"
WORKSPACE="/path/to/this/repo"
ETH_IF="enp1s0f1np1"        # CX7 ethernet interface
IB_HCA="rocep1s0f1"         # CX7 RoCE device
```

Find your interface names:
```bash
ip addr show | grep 169.254          # ethernet interface
rdma link show | grep ACTIVE         # RoCE device
```

### 3. Run

```bash
./run_dual_node.sh
```

What happens:
1. Scripts synced to worker node via rsync
2. Worker container (rank 1) started on slave
3. Head container (rank 0) started on host
4. Model weights loaded (~7 min, 46 shards streamed one by one)
5. Forward pass runs, activations extracted from target layers
6. Results saved to `activations_layer28_29.pt`

### 4. Analyze

```python
import torch
data = torch.load("activations_layer28_29.pt")
for layer_id, act in data["activations"].items():
    print(f"Layer {layer_id}: {act.shape}")  # [1, seq_len, 4, 4096]
    for stream in range(4):
        print(f"  Stream {stream} norm: {act[0, -1, stream].norm():.2f}")
```

## Files

| File | Description |
|------|-------------|
| `kernel_sm121.py` | Triton + PyTorch replacements for 6 TileLang kernels |
| `weight_loader.py` | HF shard streaming loader with key mapping + TP sharding |
| `fast_hadamard_transform.py` | Pure PyTorch Hadamard transform fallback |
| `test_dual_node.py` | Main script: model loading, forward pass, activation hooks |
| `run_dual_node.sh` | Dual-node launch script with NCCL configuration |
| `DevHistory.md` | Development log documenting all 21 pitfalls encountered |

For a detailed explanation of every change and why it was necessary, see [WHAT_WE_CHANGED.md](WHAT_WE_CHANGED.md).

## Customization

### Different layers

```python
target_layers = [28, 29]  # change to any layer indices (0-42)
```

### Different hook positions

```python
# Block.forward(x, start_pos, input_ids):
residual = x                          # input residual (4 HC streams)
x = self.hc_pre(...)                  # after HC pre-mixing (1 stream)
x = self.attn_norm(x)                 # after attention norm
x = self.attn(x, start_pos)           # attention output
x = self.hc_post(x, residual, ...)    # after HC post-mixing (4 streams)
# same pattern for FFN
```

## Who Can Use This

- **DGX Spark users**: full solution, clone and run
- **H100 / B200 users**: skip `kernel_sm121.py`, use official `kernel.py` — `weight_loader.py` and the hook framework work on any hardware
- **Anyone studying LLM internals**: the FP4 manual unpacking and streaming weight loader are reusable techniques, not DGX Spark specific

Even with a B200, you can't extract intermediate activations from a 280B model — vLLM doesn't expose hidden states, and the official `generate.py` only outputs final tokens. This platform solves that problem regardless of hardware.

## Limitations

- **Not a serving solution**: ~4 tok/s generation. Fine for research, not for production.
- **Memory-tight**: 81.8GB used of 128GB per node (DGX Spark).
- **DGX Spark kernel replacements**: `kernel_sm121.py` is specific to sm_121; on datacenter GPUs, use the official `kernel.py` instead.

## Acknowledgments

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — Docker infrastructure and custom NCCL
- [jasl/vllm](https://github.com/jasl/vllm) ds4-sm120 branch — SM120 Triton fallback approach (studied but reimplemented independently)
- DeepSeek official `inference/` code — architecture and weight format reference

## Related

- [deepseek-v4-deployment-on-dgx-spark](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark) — vLLM inference server (for serving, not research)

## License

MIT
