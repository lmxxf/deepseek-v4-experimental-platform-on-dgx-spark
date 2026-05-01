# DeepSeek V4 Flash 双机实验平台开发记录

两台 DGX Spark (128GB×2) 上搭建 DeepSeek V4 Flash (280B) 的**研究实验环境**——不是推理服务，是能提取中间层激活值、做情绪向量实验的平台。

前置项目：[deepseek-v4-flash-deployment](../deepseek-v4-flash-deployment/DevHistory.md)（推理服务已跑通）

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

## 下一步：验证 TileLang + sm_121

### 计划
1. 进现有容器，`pip install tilelang`
2. 单机试跑官方 `inference/generate.py`
3. 确认 TileLang JIT 能在 sm_121 上编译 kernel
4. 如果能跑 → 直接在官方代码基础上加 hook
5. 如果不能 → 走 jasl Triton fallback 移植路线

### 风险
- TileLang JIT 不认识 sm_121（最大风险）
- 权重格式：官方代码要求 `model{rank}-mp{world_size}.safetensors`，我们下载的可能是单文件格式
- 容器里没有 tilelang，pip install 可能有依赖冲突

---

*最后更新：2026-05-01 14:30*
