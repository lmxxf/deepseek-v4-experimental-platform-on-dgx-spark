# DeepSeek V4 Flash 双机实验平台（DGX Spark）

**在两台 DGX Spark 上提取 DeepSeek V4 Flash（280B）的中间层激活值。**

[English](README.md)

Triton + PyTorch 实现——不依赖 TileLang、DeepGEMM、vLLM。可以在 280B 模型的任意层插 hook，提取 4 路 Hyper-Connection 残差流、注意力输出、MoE 路由决策等中间状态。

## 为什么需要这个

vLLM 是黑箱推理服务器。它把模型加载、张量并行、KV cache 管理全封装了，对外只暴露 HTTP API。你没法在 forward pass 中间插 hook 提取 hidden states——就像你没法在 nginx 里面插断点看 HTTP 解析的中间状态。

本平台让你直接控制 280B 模型的内部：
- 在任意 Transformer 层提取 4 路残差流（Hyper-Connection）
- 观察注意力输出、MoE 专家路由、Sinkhorn 归一化矩阵
- 对比不同 prompt 下的激活值差异
- 用于可解释性研究、情绪向量分析、SAE 等实验

## 解决了什么问题

### SM_120 生态空白

DGX Spark 使用 GB10 GPU（sm_121），是消费级 Blackwell 架构的变体。**所有大模型工业级推理组件都不支持 sm_120 系列。**

这不是因为 sm_121 架构特殊——而是因为 sm_120 是消费级硬件（RTX 5090 等），没有人拿两张 5090 跑 280B 的分布式推理。没有市场，就没有人做基础设施。TileLang、DeepGEMM、FlashAttention 全都只支持数据中心 Blackwell（sm_100）。

DGX Spark 是第一个用消费级架构干数据中心活的产品，撞上了这个生态空白。

我们用 Triton + PyTorch 替换了所有 GPU kernel：

| 原始实现（TileLang） | 我们的替换 | 说明 |
|---|---|---|
| `act_quant` | PyTorch absmax→scale→clamp→cast | 逐块 FP8 量化 |
| `fp4_act_quant` | PyTorch FP4 查表模拟量化 | 逐块 FP4 量化 |
| `fp8_gemm` | Triton `tl.dot` + per-block scaling | FP8×FP8 矩阵乘法 |
| `fp4_gemm` | Triton `tl.dot_scaled("e4m3","e2m1")` | FP8×FP4 矩阵乘法，比手动解包快 6-7 倍 |
| `sparse_attn` | PyTorch `torch.einsum` + gather 向量化 | 稀疏注意力，无 Python 循环 |
| `hc_split_sinkhorn` | PyTorch sigmoid + softmax + Sinkhorn（5 次迭代） | Hyper-Connection 混合 |
| `fast_hadamard_transform` | PyTorch Hadamard 矩阵乘法 | CUDA kernel 在 sm_121 上编译失败 |

### 统一内存加载问题

DGX Spark 的 CPU 和 GPU 共享同一块 128GB 物理内存。这带来了独特的工程挑战：

- mmap 一个 77GB 的权重文件会 OOM：mmap 活跃页和 GPU 分配竞争同一个内存池
- 不能先在 GPU 上创建模型骨架再从 CPU 加载权重（double-buffering 会撑爆内存）
- 即使 `safe_open(device="cuda")` 内部仍会走 CPU 中转

我们的方案：从 HuggingFace 原始的 46 个 shard（每个约 3.5GB）流式加载权重，每个 shard 独立打开/处理/关闭。峰值 mmap 开销从 77GB 降到 3.5GB。这和 vLLM 内部使用的策略相同。

## 结果

```
Prefill: 9.0 秒 / 971 token（108 tok/s）
生成：153 token / 37.2 秒（4.11 tok/s）

激活值提取（第 28 层，梦境 SVG prompt）：
  Shape: [1, 971, 4, 4096]（batch, 序列长度, 4 路 HC 残差流, 隐藏维度）
  4 路 norm: [108.00, 106.50, 57.75, 30.38]  ← 高度不对称
```

数值正确——输出与 vLLM 推理服务一致。做实验够用：提取激活值只需要跑一次 forward pass。

## 硬件要求

- 2 台 NVIDIA DGX Spark（每台 128GB 统一内存）
- ConnectX-7 200Gbps 直连（QSFP56 DAC 线缆）
- DeepSeek V4 Flash 权重（约 158GB，从 HuggingFace 下载）

## 前置条件

1. 两台 DGX Spark 通过 CX7 组网，SSH 免密登录
2. Docker 镜像 `vllm-node-sm120`（来自 [deepseek-v4-deployment-on-dgx-spark](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark)），或任何包含以下组件的 Docker 镜像：
   - PyTorch 2.11+，CUDA 13.x，支持 sm_120
   - 针对 DGX Spark CX7 定制的 NCCL（来自 [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)）
   - `transformers`、`safetensors`
3. 两台机器上都下载好 DeepSeek V4 Flash 权重：
   ```bash
   huggingface-cli download deepseek-ai/DeepSeek-V4-Flash --local-dir /path/to/deepseek-v4-flash
   rsync -avP /path/to/deepseek-v4-flash user@worker-ip:/path/to/deepseek-v4-flash
   ```

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/lmxxf/deepseek-v4-experimental-platform-on-dgx-spark.git
cd deepseek-v4-experimental-platform-on-dgx-spark
```

### 2. 修改 `run_dual_node.sh`

根据你的环境修改 IP 和路径：

```bash
HEAD_IP="169.254.248.35"    # 主机 CX7 IP
WORKER_IP="169.254.30.81"   # 从机 CX7 IP
WORKER_SSH="lmxxf@169.254.30.81"
MODEL_DIR="/path/to/deepseek-v4-flash"      # 模型权重目录
WORKSPACE="/path/to/this/repo"              # 本仓库目录
ETH_IF="enp1s0f1np1"        # CX7 以太网接口名
IB_HCA="rocep1s0f1"         # CX7 RoCE 设备名
```

查找接口名：
```bash
ip addr show | grep 169.254          # 以太网接口
rdma link show | grep ACTIVE         # RoCE 设备
```

### 3. 运行

```bash
./run_dual_node.sh
```

脚本会自动完成以下步骤：
1. 通过 rsync 同步脚本到从机
2. 在从机上启动 worker 容器（rank 1）
3. 在主机上启动 head 容器（rank 0）
4. 流式加载模型权重（约 7 分钟，46 个 shard 逐个加载）
5. 运行 forward pass，从目标层提取激活值
6. 保存结果到 `activations_layer28_29.pt`

### 4. 分析激活值

```python
import torch
data = torch.load("activations_layer28_29.pt")
for layer_id, act in data["activations"].items():
    print(f"Layer {layer_id}: {act.shape}")  # [1, seq_len, 4, 4096]
    for stream in range(4):
        print(f"  Stream {stream} norm: {act[0, -1, stream].norm():.2f}")
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `kernel_sm121.py` | 6 个 TileLang kernel 的 Triton + PyTorch 替换实现 |
| `weight_loader.py` | HF shard 流式加载器（含 key 名映射 + TP 分片） |
| `fast_hadamard_transform.py` | Hadamard 变换的纯 PyTorch 替换 |
| `test_dual_node.py` | 主脚本：模型加载、forward pass、激活值 hook |
| `run_dual_node.sh` | 双机启动脚本（含 NCCL 配置 + rsync 同步） |
| `DevHistory.md` | 开发日志，记录了全部 21 个踩坑 |

每个改动的详细说明和原因，参见 [WHAT_WE_CHANGED_zh.md](WHAT_WE_CHANGED_zh.md)。

## 自定义

### 提取不同的层

在 `test_dual_node.py` 中修改：
```python
target_layers = [28, 29]  # 改为任意层号（0-42）
```

### Hook 不同位置

`Block.forward()` 中可以 hook 的位置：
```python
# Block.forward(x, start_pos, input_ids):
residual = x                          # 输入残差流（4 路 HC）
x = self.hc_pre(...)                  # HC 前混合后（1 路）
x = self.attn_norm(x)                 # 注意力归一化后
x = self.attn(x, start_pos)           # 注意力输出
x = self.hc_post(x, residual, ...)    # HC 后混合后（4 路）
# FFN 部分同样模式
```

## 谁能用

- **有 DGX Spark 的用户**：完整方案，克隆即用
- **有 H100 / B200 的用户**：跳过 `kernel_sm121.py`，用官方 `kernel.py`，`weight_loader.py` + hook 框架直接可用
- **想学习 LLM 内部机制的人**：FP4 手动解包和流式权重加载是通用技术，不限于 DGX Spark

即使你有 B200，你也做不到从 280B 模型里提取中间层激活值——vLLM 不暴露 hidden state，官方 `generate.py` 只输出最终 token。本平台解决的是这个问题，跟硬件无关。

## 局限性

- **不是推理服务**：生成约 4 tok/s。做研究够用，不能当生产环境。
- **内存紧张**：每台 128GB 中用了 81.8GB（DGX Spark）。
- **DGX Spark kernel 替换**：`kernel_sm121.py` 是针对 sm_121 的；数据中心 GPU 上直接用官方 `kernel.py`。

## 致谢

- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — Docker 基础设施和 DGX Spark 定制 NCCL
- [jasl/vllm](https://github.com/jasl/vllm) ds4-sm120 分支 — SM120 Triton fallback 方案（我们研究了他的方法，独立实现了纯 PyTorch 版本）
- DeepSeek 官方 `inference/` 代码 — 模型架构和权重格式参考

## 相关项目

- [deepseek-v4-deployment-on-dgx-spark](https://github.com/lmxxf/deepseek-v4-deployment-on-dgx-spark) — 基于 vLLM 的推理服务（用于部署，不用于研究）

## 许可证

MIT
