# DeepSeek V4 Flash 双机实验平台开发记录

两台 DGX Spark (128GB×2) 上搭建 DeepSeek V4 Flash (280B) 的**研究实验环境**——不是推理服务，是能提取中间层激活值、做情绪向量实验的平台。

前置项目：[deepseek-v4-flash-deployment](../deepseek-v4-flash-deployment/DevHistory.md)（推理服务已跑通）

---

## 核心认知：sm_120 系列没有多卡生态

**这不是 sm_121 孤儿的问题，是 sm_120 整个系列都没有大模型分布式推理的基础设施。**

sm_120 是消费级 Blackwell（RTX 5090 等），使用场景是打游戏和单卡推理。没有人拿两张 5090 跑 280B 的分布式推理——没有市场就没有人做基础设施。所以 DeepGEMM、TileLang、vLLM 官方全都只支持 sm_100（数据中心 Blackwell，给 H100/B200 集群用的），不支持 sm_120。

DGX Spark 用了 sm_121（sm_120 的超集变体），却要干数据中心的活（双机 256GB 跑 280B），撞上了这个生态空白。eugr 的定制 NCCL、jasl 的 Triton fallback、我们现在做的实验平台，本质上都是在**给一个"不应该存在的使用场景"补基础设施**。

**所有大模型的工业级推理组件（DeepGEMM、TileLang、FlashAttention、vLLM 原生）都不会支持 sm_120。** 我们做的东西以 sm_120 为基础，但实际上只有 DGX Spark 用得到。

（注：两块 RTX 5090 32GB×2 通过 PCIe 高速推理 70B 4bit 量化模型是可行的，但那是单机多卡场景，PCIe 带宽够用，不需要 RDMA/NVLink 级别的分布式基础设施。和我们的 280B 双机场景完全不同。）

---

## 背景：为什么需要新平台

### 现有推理服务的局限
- vLLM 是黑箱推理服务器，不暴露中间层激活值
- 所有 TP 分片、KV cache、调度全封装在 vLLM 内部
- 无法在 forward pass 中插 hook 提取 hidden states
- 无法做开灯/关灯对比实验、情绪向量分析等研究

### 目标
- 纯 PyTorch 环境，双机 TP=2 加载 280B 模型
- 在任意层插 `register_forward_hook()` 提取激活值
- 支持保存中间层激活到 .pt 文件，做离线分析
- 支持 PCA 降维可视化、余弦相似度对比等

---

## 第一阶段：方案调研（2026-05-01）

### 初始假设（已推翻）
原以为 DeepSeek V4 官方推理代码 (`inference/model.py`) 直接调用 DeepGEMM，需要把 jasl vLLM fork 的 2000 行 Triton fallback kernel 移植过来。

### 关键发现：官方代码用 TileLang，不是 DeepGEMM

读完 `deepseek-v4-flash/inference/` 三个文件后发现：

| 组件 | vLLM 里 | 官方 inference/ 里 |
|------|---------|-------------------|
| GEMM kernel | DeepGEMM → jasl Triton fallback | **TileLang JIT** |
| Hyperconnection | DeepGEMM `tf32_hc_prenorm_gemm` | **TileLang `hc_split_sinkhorn`** |
| 注意力 | DeepGEMM MQA + vLLM paged cache | **TileLang `sparse_attn`** |
| FP8/FP4 量化 | DeepGEMM einsum | **TileLang `fp8_gemm` / `fp4_gemm`** |

**工作量从"移植 2000 行 Triton kernel"缩减到"装个 tilelang 然后试一下"。**

### 官方推理代码结构（inference/）

**model.py（38KB）—— 模型定义**
- 自带 TP 实现：`ParallelEmbedding`、`ColumnParallelLinear`、`RowParallelLinear`
- 通过全局变量 `world_size`、`rank` 管理分布式
- MLA 注意力、256 路由专家 MoE、Hyper-Connections 全在里面
- 所有 kernel 来自 `kernel.py`，不依赖 DeepGEMM

**generate.py —— 分布式推理入口**
- `torch.distributed` + NCCL 初始化
- 环境变量 `WORLD_SIZE`、`RANK`、`LOCAL_RANK` 控制
- 按 rank 加载对应分片权重 `model{rank}-mp{world_size}.safetensors`
- 支持交互模式和批量模式

**kernel.py（22KB）—— TileLang 自定义 kernel**
- 6 个 kernel：`act_quant`、`fp4_act_quant`、`fp8_gemm`、`fp4_gemm`、`sparse_attn`、`hc_split_sinkhorn`
- 全部基于 TileLang JIT（`@tilelang.jit` 装饰器）
- 无硬编码 CUDA 架构检查，理论上架构无关

### Hook 插入点（model.py）
- 每个 Block 的 `forward()` 里：
  - `hc_pre` 之后（pre-attention hidden state）
  - attention 输出之后
  - `hc_pre` 之后（pre-FFN hidden state）
  - FFN 输出之后
- 最终 head 投影之前（所有 block 走完后）

### jasl Triton fallback 分析（备用方案）

如果 TileLang 在 sm_121 不能用，可以从 jasl fork 提取 Triton kernel：

| Kernel | 可提取性 | 原因 |
|--------|---------|------|
| `tf32_hc_prenorm_gemm_triton` | ✅ 容易 | 纯 torch + triton，无 vLLM 依赖 |
| `deepseek_v4_fp8_einsum_triton` | ✅ 容易 | 纯计算 |
| `deepseek_v4_sm12_fp8_einsum` | ✅ 容易 | 70 行 |
| `fp8_mqa_logits_triton` | ⚠️ 中等 | 需替换 vllm.triton_utils 导入 |
| `fp8_paged_mqa_logits_triton` | ❌ 困难 | 深度耦合 vLLM paging 系统 |

### 现有基础设施

- **容器**：`vllm-node-sm120`（23.9GB），内含 PyTorch 2.11.0 + Transformers 5.6.2 + Ray + 定制 NCCL
- **网络**：CX7 200Gbps 直连，SSH 免密
- **权重**：两台都有 `/home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash/`（158GB）
- **内存预算**：总 206GB 可用，模型 148GB，剩余 ~58GB 做实验

---

## 第二阶段：TileLang sm_121 验证（2026-05-01）

### 结论：TileLang 0.1.8 不支持 sm_121 ❌

TileLang v0.1.8（DeepSeek 官方指定版本）明确支持的架构：
- SM75 (Turing)、SM90 (Hopper)、SM100 (数据中心 Blackwell)

**不支持 SM_120/SM_121**。和 DeepGEMM 一样的问题——sm_121 孤儿架构。

路线修正：不再尝试 TileLang，直接写 `kernel_sm121.py` 替换 `kernel.py`。

---

## 第三阶段：kernel 替换方案设计（2026-05-01）

### 核心思路
写一个 `kernel_sm121.py`，用 Triton + 纯 PyTorch 重新实现 `kernel.py` 的 6 个函数接口，让 `model.py` 只需要改一行 import。

### 6 个 kernel 对照表

| # | 函数名 | 功能 | 输入→输出 | 替换策略 | 难度 |
|---|--------|------|----------|---------|------|
| 1 | `act_quant(x, block_size, scale_fmt, scale_dtype, inplace)` | FP8 逐块量化 | BF16→FP8+scale | 纯 PyTorch：absmax→scale→clamp→cast | 低 |
| 2 | `fp4_act_quant(x, block_size, inplace)` | FP4 逐块量化 | BF16→FP4+scale | 纯 PyTorch：同上换 clamp 范围 | 低 |
| 3 | `fp8_gemm(a, a_s, b, b_s, scale_dtype)` | FP8×FP8 GEMM | FP8 A[M,K] × FP8 B[N,K]^T → BF16 C[M,N] | `torch._scaled_mm` 或 Triton | 中 |
| 4 | `fp4_gemm(a, a_s, b, b_s, scale_dtype)` | FP8×FP4 GEMM | FP8 act × FP4 weight → BF16 | cast FP4→BF16 再 matmul（慢）或 Triton | 中 |
| 5 | `sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)` | 稀疏注意力 | Q[b,s,h,d] + KV[b,n,d] + idxs → O[b,s,h,d] | 纯 PyTorch index_select + bmm + online softmax | 高 |
| 6 | `hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)` | Hyper-Connection | mixes→pre,post,comb | 纯 PyTorch：sigmoid + softmax + 迭代归一化 | 低 |

### 策略：先跑通再优化

**Phase A（纯 PyTorch fallback）**：全部 6 个 kernel 用纯 PyTorch 实现
- 速度很慢，但能验证模型加载、双机通信、hook 机制
- 预计推理一个 token 可能需要几秒甚至十几秒
- **够用**——实验只需要跑几十个 token 拿激活值，不是做推理服务

**Phase B（Triton 加速）**：把热点 kernel 换成 Triton
- 优先：`fp8_gemm`、`fp4_gemm`（占推理时间最大）
- 其次：`sparse_attn`（注意力计算）
- 最后：量化和 HC（占比小）

### jasl 可复用的部分

jasl 的 Triton kernel 虽然嵌在 vLLM 里，但以下函数接口干净，可以直接抄逻辑：
- `tf32_hc_prenorm_gemm_triton` → 对应我们的 `hc_split_sinkhorn`（功能不完全一样，但 Sinkhorn 部分可参考）
- `deepseek_v4_fp8_einsum_triton` → FP8 GEMM 的 Triton 实现模式可参考
- scale 格式转换辅助函数（`_e8m0_to_fp32`、`_unpack_int32_e8m0_scales`）→ 直接复用

### 待确认

1. **权重格式**：官方代码要 `model{rank}-mp{world_size}.safetensors`（TP 分片权重），我们下载的是 HF 格式（46 个 shard）。可能需要跑转换脚本（`inference/convert.py`？）
2. **`torch._scaled_mm` 在 sm_121 上的行为**：PyTorch 2.11 的 FP8 GEMM 支持到哪个架构？
3. **FP4 tensor 类型**：`torch.float4_e2m1fn_x2` 是 PyTorch 2.11 新增的，sm_121 上能不能用？
4. **容器选择**：用现有 `vllm-node-sm120`（已有定制 NCCL + PyTorch），还是新建容器？

---

## 下一步

1. 进容器确认 `torch._scaled_mm` 和 FP4 类型在 sm_121 上能用
2. 检查权重格式，确认是否需要转换
3. 开始写 `kernel_sm121.py` Phase A（纯 PyTorch fallback）
4. 单机测试加载模型 + forward pass

---

*最后更新：2026-05-01 15:00*
