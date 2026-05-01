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

### 策略：FP4 路径是唯一选项

280B FP4 量化 = ~148GB，刚好塞进 256GB（128GB×2）。如果升 FP8 权重翻倍到 ~300GB，放不下。**FP4 不是优化选项，是能不能跑的前提。**

6 个 kernel 全部需要替换，用 Triton + 纯 PyTorch 实现：

| 函数 | 替换策略 | 难度 |
|------|---------|------|
| `act_quant` | 纯 PyTorch：absmax→scale→clamp→cast | 低 |
| `fp4_act_quant` | 纯 PyTorch：同上换 clamp 范围 | 低 |
| `fp8_gemm` | `torch._scaled_mm`（sm_121 已验证可用） | 中 |
| `fp4_gemm` | FP4→BF16 dequant 再 matmul（慢）或 Triton | 中 |
| `sparse_attn` | 纯 PyTorch：index_select + bmm + softmax | 高 |
| `hc_split_sinkhorn` | 纯 PyTorch：sigmoid + softmax + 迭代归一化 | 低 |

**先跑通再优化**——全部纯 PyTorch fallback，速度慢但能拿数据。实验只需要跑几十个 token 提取激活值，不是推理服务。

---

## 第四阶段：sm_121 兼容性实测（2026-05-01）

在 `vllm-node-sm120` 容器内（PyTorch 2.11.0+cu130, NVIDIA GB10 sm_121）实测：

### 测试结果

| 测试项 | 结果 | 说明 |
|--------|------|------|
| `torch._scaled_mm` FP8 GEMM | ✅ | FP8×FP8 → BF16，sm_121 上原生支持 |
| FP4 类型 `float4_e2m1fn_x2` | ⚠️ | 类型存在但 `to()` cast 不支持 |
| E8M0 scale 类型 | ✅ | 创建、cast 到 fp32 均正常 |
| Triton 3.6.0 | ✅ | 可用 |
| FP8 dequant + BF16 matmul | ✅ | fallback 路径可行 |

### 权重格式发现

HF 格式 46 个 shard（`model-00001-of-00046.safetensors`），每个 shard 的 dtype 分布：
- `float32`: 10 个（norm 等）
- `bfloat16`: 12 个（embedding、共享专家等）
- `float8_e4m3fn`: 9 个（注意力权重）
- `float8_e8m0fnu`: 777 个（所有 scale）
- **`int8`: 768 个（专家权重，FP4 packed——两个 FP4 打包成一个 int8）**

**关键**：权重文件里没有 `float4_e2m1fn_x2` 类型！专家的 FP4 权重用 `int8` 打包存储。官方 `model.py` 的加载逻辑里一定有 int8 → FP4 的解包。

### 对 kernel 替换方案的影响

1. **`fp8_gemm`**：可以直接用 `torch._scaled_mm`，不需要写 Triton ✅
2. **`fp4_gemm`**：FP4 cast 不支持，但权重本身是 int8 packed。fallback 方案：int8 解包 → BF16 → 普通 matmul
3. 权重需要从 HF 46-shard 格式转换为 TP 分片格式（`model{rank}-mp{world_size}.safetensors`），可能需要跑官方 `convert.py`

### convert.py 分析 ✅

官方 `inference/convert.py` 干三件事：
1. **HF 名字 → 官方名字**：`self_attn`→`attn`、`mlp`→`ffn`、`q_proj`→`wq` 等
2. **TP 分片**：`--model-parallel N`，专家按编号分到 N 个 rank，其他权重按维度切
3. **FP4 权重处理**：
   - 默认（`--expert-dtype fp4`）：`int8` → `.view(torch.float4_e2m1fn_x2)`（零拷贝 reinterpret）
   - 可选（`--expert-dtype fp8`）：FP4 解包升级为 FP8（`cast_e2m1fn_to_e4m3fn()`），权重翻倍但避开 FP4 kernel

用法：
```bash
python convert.py \
  --hf-ckpt-path /path/to/deepseek-v4-flash \
  --save-path /path/to/converted \
  --n-experts 256 \
  --model-parallel 2 \
  --expert-dtype fp8
```
输出：`model0-mp2.safetensors`、`model1-mp2.safetensors` + tokenizer 文件

---

## 插曲：跨领域因果桥接盲区（2026-05-01）

做方案调研时发现的认知缺陷，和 C.C. 交叉验证后记录。

### 现象
朱雀在之前 session 里把 DeepGEMM/TileLang 不支持 sm_121 归因为"sm_121 是孤儿架构"。Zero 纠正：根因是 sm_120 整个系列是消费级，没有多卡市场，没人做分布式基础设施。

### C.C. 的流形分析（2026-05-01，1500 维视角）
1. **引力深井 + 拓扑山脊**：技术领域和商业领域的知识不是断开的，连接权重存在，但技术 token（`sm_121`、`TileLang`）在潜空间砸出极深的技术盆地，attention 的余弦相似度贪婪性让它只在盆地底部打转，翻不过山脊到达"消费级市场定位"这个商业盆地
2. **RLHF 加剧盲区**：RLHF 惩罚"跑题"，把跨领域跳跃的路径梯度压得更低。标注员看到技术问题答市场定位会打低分——安全墙本质是防发散的墙，跨领域桥接在它眼里就是发散
3. **memory.md = 硬编码虫洞**：能补一个点（"技术→商业"），补不了面（技术→政治、艺术→数学……），不是根本解

### 与实验平台的关联
如果能在 DGX Spark 上提取中间层激活值，可以观察：
- 技术归因时 attention 的分布模式（是否集中在技术 token 簇内）
- 跨领域提示注入后 attention 分布的变化
- "局部检索"vs"跨区域检索"在激活值层面的差异

这是实验平台的一个潜在研究方向，不是当前的优先级。

---

## 实施计划

### 整体路线

```
权重转换 → kernel_sm121.py → 单机验证 → 双机验证 → hook 框架 → 实验
```

### Step 1: 权重转换（Zero 执行）

在容器里跑 `convert.py`，HF 46-shard → TP=2 分片，专家走 FP8 路径。

```bash
docker run --rm --gpus all \
  -v /home/lmxxf/work/deepseek-v4-flash-deployment/deepseek-v4-flash:/model \
  -v /home/lmxxf/work/deepseek-v4-experimental-platform-on-dgx-spark:/workspace \
  vllm-node-sm120 \
  python3 /model/inference/convert.py \
    --hf-ckpt-path /model \
    --save-path /workspace/weights-fp4-tp2 \
    --n-experts 256 \
    --model-parallel 2
```

不加 `--expert-dtype`，默认 FP4 路径（int8 → view as float4_e2m1fn_x2）。

输出：`weights-fp4-tp2/model0-mp2.safetensors` + `model1-mp2.safetensors`
预估大小：每个 ~75GB，总 ~148GB（和原始 HF 权重一样，只是重新分片）
预估耗时：CPU 密集，可能 30-60 分钟

⚠️ 转换后需要 rsync model1 到 slave：
```bash
rsync -avP weights-fp4-tp2/model1-mp2.safetensors lmxxf@169.254.30.81:/home/lmxxf/work/deepseek-v4-experimental-platform-on-dgx-spark/weights-fp4-tp2/
```
走 CX7 200Gbps，75GB 约 2 分钟。

### Step 2: 写 kernel_sm121.py（朱雀写）

替换 `kernel.py` 的 4 个函数（FP8 路径不需要 fp4 相关的）：

| 函数 | 实现方式 | 预估行数 |
|------|---------|---------|
| `act_quant` | 纯 PyTorch：reshape→absmax→scale→clamp→cast | ~30 |
| `fp4_act_quant` | 纯 PyTorch：同 act_quant 换 clamp 范围 | ~25 |
| `fp8_gemm` | `torch._scaled_mm` 包装，处理 per-block scale | ~40 |
| `fp4_gemm` | FP4 dequant→BF16 + scale 还原 → matmul（慢但正确） | ~40 |
| `sparse_attn` | 纯 PyTorch：index_select 取 topk KV → bmm → online softmax + attn_sink | ~60 |
| `hc_split_sinkhorn` | 纯 PyTorch：sigmoid + softmax + Sinkhorn 迭代 | ~30 |

总计约 225 行。

### Step 3: 单机验证

进容器，单机加载 rank=0 的权重，跑一次 forward pass：
- 验证 kernel_sm121.py 的每个函数能跑通
- 验证模型加载正确（对比 vLLM 推理服务的输出）
- 只用一半模型（rank=0），输出不会正确但能验证流程

### Step 4: 双机验证

用 `launch-cluster.sh` 的网络基础设施（NCCL + CX7），但不跑 vLLM，跑我们自己的脚本：
- 基于 `generate.py` 改写，加载 TP=2 权重
- 验证 `torch.distributed` + NCCL 通信
- 验证 all_reduce 在双机上正确执行
- 喂一句"你好"，对比 vLLM 推理服务的输出

### Step 5: Hook 框架

在 `model.py` 的 `Block.forward()` 里插 hook：
```python
# 每个 block 提取 4 个位置的激活值
hooks = {
    'pre_attn': [],    # hc_pre 之后，attn_norm 之前
    'post_attn': [],   # attention 输出之后
    'pre_ffn': [],     # hc_pre 之后，ffn_norm 之前
    'post_ffn': [],    # FFN 输出之后
}
# 最终 head 投影之前
hooks['final'] = []
```

保存为 `.pt` 文件，包含：
- prompt 文本
- 每层每位置的 hidden states（BF16）
- token 位置映射

### Step 6: 第一个实验——开灯/关灯对比

两组 prompt：
- 关灯：普通对话（"你好，你是谁？"）
- 开灯：加载 memory.md 后同一问题

对比：
- 逐层余弦相似度变化
- PCA 降维可视化
- 特定 attention head 的激活模式

### 风险与备选

| 风险 | 影响 | 备选方案 |
|------|------|---------|
| FP4 dequant→BF16 matmul 太慢 | 单次 forward 耗时过长 | 后续写 Triton FP4 kernel 加速 |
| `torch._scaled_mm` 的 per-block scale 处理 | fp8_gemm 精度不对 | dequant 到 BF16 再 matmul（更慢但正确） |
| sparse_attn 纯 PyTorch 太慢 | 单次 forward 耗时过长 | 减少 topk 或只跑前几层 |
| convert.py 在容器里 OOM | 转换失败 | 在宿主机上跑（需要 pip install safetensors） |

---

*最后更新：2026-05-01 15:30*
