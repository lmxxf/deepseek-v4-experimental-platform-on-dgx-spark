# DeepSeek V4 Flash Experimental Platform on DGX Spark

**Extract intermediate activations from DeepSeek V4 Flash (280B) on two DGX Spark nodes.**

Pure PyTorch implementation — no TileLang, no DeepGEMM, no vLLM. Direct access to every layer's residual stream, attention outputs, and MoE routing decisions.

在两台 DGX Spark（128GB×2）上运行 DeepSeek V4 Flash 280B 的研究实验平台。纯 PyTorch 实现，可以提取任意层的中间层激活值（4 路 Hyper-Connection 残差流）。

## Why / 为什么需要这个

vLLM is a black-box inference server. You can't hook into the forward pass to extract hidden states, attention patterns, or MoE routing decisions. This platform gives you full control over the model's internals.

vLLM 是黑箱推理服务器，无法提取中间层激活值。本平台让你可以在 280B 模型的任意层插 hook，dump 残差流、注意力输出、专家路由等中间状态。

## What we solved / 解决了什么问题

DGX Spark uses GB10 GPUs (sm_121), a consumer-grade Blackwell variant. **No major ML framework supports sm_120-series for distributed inference** — there's no market for multi-GPU setups on consumer hardware, so nobody built the infrastructure.

DGX Spark 使用 GB10 GPU（sm_121），属于消费级 Blackwell 架构。所有大模型工业级推理组件（TileLang、DeepGEMM、FlashAttention）都不支持 sm_120 系列——因为消费级硬件没有多卡推理市场，没人做。

We replaced every GPU kernel with pure PyTorch fallbacks:

| Original (TileLang) | Our Replacement | Notes |
|---|---|---|
| `act_quant` | PyTorch absmax→scale→clamp→cast | Block-wise FP8 quantization |
| `fp4_act_quant` | PyTorch FP4 simulation via lookup table | Block-wise FP4 quantization |
| `fp8_gemm` | PyTorch dequant→BF16→`torch.mm` | FP8×FP8 with per-block scaling |
| `fp4_gemm` | Manual FP4 unpack (int8→FP4_TABLE)→BF16→matmul | `float4_e2m1fn_x2.to()` doesn't work on sm_121 |
| `sparse_attn` | PyTorch for-loop + bmm + softmax | Sparse attention with index gathering |
| `hc_split_sinkhorn` | PyTorch sigmoid + softmax + Sinkhorn iteration | Hyper-Connection mixing |
| `fast_hadamard_transform` | PyTorch Hadamard matrix multiply | CUDA kernel won't compile on sm_121 |

We also solved the unified memory loading problem: DGX Spark shares 128GB between CPU and GPU. Loading a 77GB weight file via mmap causes OOM because mmap pages and GPU allocations compete for the same physical memory pool. Our solution: stream weights from 46 small HuggingFace shards (~3.5GB each), same approach as vLLM.

另外解决了统一内存加载问题：DGX Spark 的 CPU 和 GPU 共享 128GB 内存，mmap 大文件会和 GPU 分配竞争同一块物理内存导致 OOM。我们的方案：从 HuggingFace 原始 46 个 shard（每个约 3.5GB）流式加载权重，和 vLLM 使用相同的策略。

## Results / 结果

```
Forward pass OK! 5.5s (prefill, 12 tokens)
Generation: 29 tokens in 52.7s (~1.8s/token)
Output: "你好！我是DeepSeek，一个由深度求索公司创造的AI助手，乐于为你解答问题、提供帮助和分享知识。"

Activation extraction:
  Layer 28 residual: [1, 12, 4, 4096]
  4-stream norms: [133.00, 166.00, 113.50, 26.88]  ← 4 Hyper-Connection streams, highly asymmetric
```

Slow (pure PyTorch, no Triton optimization), but correct — output matches vLLM inference server. Good enough for research: you only need one forward pass to extract activations.

速度慢（纯 PyTorch，无 Triton 优化），但数值正确——输出与 vLLM 推理服务一致。做实验够用：提取激活值只需要跑一次 forward pass。

## Hardware Requirements / 硬件要求

- 2× NVIDIA DGX Spark (128GB unified memory each)
- ConnectX-7 200Gbps direct connection (QSFP56 DAC cable)
- DeepSeek V4 Flash weights (~158GB total, downloaded from HuggingFace)

## Prerequisites / 前置条件

1. Two DGX Spark nodes networked via CX7 with SSH passwordless access
2. Docker with `vllm-node-sm120` image (from [deepseek-v4-deployment-on-dgx-spark](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark)), or any Docker image with:
   - PyTorch 2.11+ with CUDA 13.x and sm_120 support
   - Custom NCCL built for DGX Spark CX7 (from [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker))
   - `transformers`, `safetensors`
3. DeepSeek V4 Flash weights downloaded to both nodes:
   ```bash
   huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir /path/to/deepseek-v4-flash
   ```

## Quick Start / 快速开始

### 1. Clone this repo on both nodes

```bash
git clone https://github.com/lmxxf/deepseek-v4-experimental-platform-on-dgx-spark.git
cd deepseek-v4-experimental-platform-on-dgx-spark
```

### 2. Edit `run_dual_node.sh`

Update paths and IPs for your setup:

```bash
HEAD_IP="169.254.248.35"    # host CX7 IP
WORKER_IP="169.254.30.81"   # slave CX7 IP
WORKER_SSH="lmxxf@169.254.30.81"
MODEL_DIR="/path/to/deepseek-v4-flash"
WORKSPACE="/path/to/this/repo"
ETH_IF="enp1s0f1np1"        # CX7 ethernet interface name
IB_HCA="rocep1s0f1"         # CX7 RoCE device name
```

Find your interface names:
```bash
ip addr show | grep 169.254          # → ethernet interface
rdma link show | grep ACTIVE         # → RoCE device
```

### 3. Run

```bash
./run_dual_node.sh
```

This will:
1. Sync scripts to the worker node
2. Start worker container (rank 1) on the slave
3. Start head container (rank 0) on the host
4. Load model weights (~7 minutes, 46 shards streamed one by one)
5. Run forward pass and extract activations from layers 28-29
6. Save activations to `activations_layer28_29.pt`

### 4. Analyze activations

```python
import torch
data = torch.load("activations_layer28_29.pt")
for layer_id, act in data["activations"].items():
    print(f"Layer {layer_id}: {act.shape}")  # [1, seq_len, 4, 4096]
    for stream in range(4):
        print(f"  Stream {stream} norm: {act[0, -1, stream].norm():.2f}")
```

## Files / 文件说明

| File | Description |
|------|-------------|
| `kernel_sm121.py` | Pure PyTorch replacements for TileLang kernels (FP4/FP8 GEMM, sparse attention, Sinkhorn) |
| `weight_loader.py` | Stream weights from HF shards with key mapping + TP sharding (no convert.py needed) |
| `fast_hadamard_transform.py` | Pure PyTorch Hadamard transform (replaces CUDA kernel) |
| `test_dual_node.py` | Main script: load model, run forward pass, extract activations |
| `run_dual_node.sh` | Launch script: start containers on both nodes with NCCL config |
| `DevHistory.md` | Development log with all 21 pitfalls documented |

## Customization / 自定义

### Extract different layers

In `test_dual_node.py`, change:
```python
target_layers = [28, 29]  # ← change to any layer indices (0-42)
```

### Hook different positions in the forward pass

The `Block.forward()` has these hookable positions:
```python
# Block.forward(x, start_pos, input_ids):
residual = x                          # ← input residual (4 streams)
x = self.hc_pre(...)                  # ← after HC pre-mixing (1 stream)
x = self.attn_norm(x)                 # ← after attention norm
x = self.attn(x, start_pos)           # ← attention output
x = self.hc_post(x, residual, ...)    # ← after HC post-mixing (4 streams)
# ... same pattern for FFN ...
```

### Run text generation instead of activation extraction

The script includes generation code (currently commented out in favor of activation extraction). See `DevHistory.md` for the generation version that produces correct output.

## Limitations / 局限性

- **Slow**: ~1.8s per token (pure PyTorch, no Triton/CUDA kernel optimization). Fine for research, not for serving.
- **Memory-tight**: 81.8GB of 128GB used per node, ~46GB free for activations and computation.
- **DGX Spark only**: The kernel replacements and weight loading strategy are specific to sm_121 + unified memory architecture.

## Acknowledgments / 致谢

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — Docker infrastructure and custom NCCL for DGX Spark
- [jasl/vllm](https://github.com/jasl/vllm) ds4-sm120 branch — Triton fallback kernels for SM120 (we studied their approach but implemented our own pure PyTorch version)
- DeepSeek official `inference/` code — model architecture and weight format reference

## Related / 相关项目

- [deepseek-v4-deployment-on-dgx-spark](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark) — vLLM-based inference server (for serving, not research)

## License

MIT
