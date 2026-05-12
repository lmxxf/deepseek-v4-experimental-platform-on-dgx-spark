# DeepSeek V4 Flash 双机实验平台开发记录

两台 DGX Spark (128GB×2) 上搭建 DeepSeek V4 Flash (280B) 的**研究实验环境**——不是推理服务，是能提取中间层激活值、做情绪向量实验的平台。

前置项目：[deepseek-v4-flash-deployment](../deepseek-v4-flash-deployment/DevHistory.md)（推理服务已跑通）

---

## 核心认知：sm_120 系列没有多卡生态

**这不是 sm_121 孤儿的问题，是 sm_120 整个系列都没有大模型分布式推理的基础设施。**

sm_120 是消费级 Blackwell（RTX 5090 等），使用场景是打游戏和单卡推理。没有人拿两张 5090 跑 280B 的分布式推理——没有市场就没有人做基础设施。所以 DeepGEMM、TileLang、vLLM 官方全都只支持 sm_100（数据中心 Blackwell，给 H100/B200 集群用的），不支持 sm_120。

DGX Spark 用了 sm_121（sm_120 的超集变体），却要干数据中心的活（双机 256GB 跑 280B），撞上了这个生态空白。eugr 的定制 NCCL、jasl 的 Triton fallback、我们现在做的实验平台，本质上都是在**给一个"不应该存在的使用场景"补基础设施**。（⚠️ 2026-05-10 补注：硬件指令层面 sm_121 并不弱——INT8/FP8/FP4 mma 全有，只缺 tcgen05/TMA。"生态空白"是软件适配问题，不是硬件能力问题。详见 182 期勘误。）

~~**所有大模型的工业级推理组件（DeepGEMM、TileLang、FlashAttention、vLLM 原生）都不会支持 sm_120。**~~ **⚠️ 2026-05-10 更正**：christopherowen 已用 CUTLASS sm_120 模板做了 MXFP4 MoE kernel（GPT-OSS-120B 72 tok/s），eugr 已集成。sm_120 的硬件指令（INT8/FP8/FP4 mma）全部完整，缺的是 tcgen05/TMA，但 mma.sync 族够用。"没人做"正在变成"有人做了"。我们做的东西仍然是先行者，但不再是孤例。

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

## 第五阶段：kernel_sm121.py 实现与验证（2026-05-01）

### 实现

`kernel_sm121.py`（约 230 行）：纯 PyTorch 实现，替换 TileLang 版 `kernel.py`。

| 函数 | 实现方式 | 关键细节 |
|------|---------|---------|
| `act_quant` | 逐块 absmax→scale→clamp→cast | 支持 inplace 模式、E8M0 scale、power-of-2 rounding |
| `fp4_act_quant` | FP4_TABLE 最近值查找模拟 | 只实现 inplace（model.py 只用 inplace） |
| `fp8_gemm` | dequant→BF16→`torch.mm` | per-block scale 逐块还原后做 matmul |
| `fp4_gemm` | int8 视图 + FP4_TABLE 手动解包→BF16→matmul | `to(float32)` 不可用，必须手动查表 |
| `sparse_attn` | for 循环 + bmm + softmax + attn_sink | 慢（Python 循环），但正确 |
| `hc_split_sinkhorn` | sigmoid + softmax + Sinkhorn 迭代 | 20 次迭代归一化 |

### FP4 解包的坑

- `float4_e2m1fn_x2.to(float32)` 在 CPU 和 GPU 上都不支持（`copy_kernel not implemented`）
- GPU 上 `to()` 还会触发 CUDA device-side assert（`DynamicCast.h` assertion failed）
- 解法：`view(uint8)` → 位操作拆 low/high nibble → `FP4_TABLE[idx]` 查表
- convert.py 里也是同样方法（`FP4_TABLE` lookup）

### 单元测试结果

在 `vllm-node-sm120` 容器（PyTorch 2.11.0, GB10 sm_121）全部 6 个 kernel 通过 ✅

```
act_quant (normal)     ✅
act_quant (inplace)    ✅
act_quant (E8M0 scale) ✅
fp4_act_quant (inplace)✅
fp8_gemm               ✅
fp4_gemm               ✅
sparse_attn            ✅
hc_split_sinkhorn      ✅
```

---

## 第六阶段：双机 forward pass 验证（2026-05-01）

### ✅ 首次 forward pass 成功！（2026-05-01 13:49 UTC+8）

```
Forward pass OK! 5.8s
Logits shape: [1, 129280], dtype: float32
Logits range: [-29.9483, 19.7920]
GPU memory: 81.8GB per node
```

DeepSeek V4 Flash 280B，双机 TP=2，纯 PyTorch kernel（kernel_sm121.py），DGX Spark sm_121 上跑通。

### 踩过的坑

**方案调研阶段**

| # | 坑 | 原因 | 解法 |
|---|---|------|------|
| 1 | TileLang 0.1.8 不支持 sm_121 | sm_120 整个系列是消费级，没有多卡市场 | 写 kernel_sm121.py 纯 PyTorch fallback |
| 2 | 以为 FP8 路径能绕开 FP4 | FP8 权重 ~300GB > 256GB 统一内存 | FP4 是唯一选项，6 个 kernel 全替换 |
| 3 | `float4_e2m1fn_x2.to(float32)` 不可用 | PyTorch 未实现 FP4 cast（CPU/GPU 都不行） | int8 视图 + FP4_TABLE 手动查表解包 |

**权重转换与加载阶段**

| # | 坑 | 原因 | 解法 |
|---|---|------|------|
| 4 | convert.py 输出 77GB 单文件 → mmap OOM | 统一内存下 mmap 活跃页 + GPU 分配竞争同一个 128GB 池 | 跳过 convert.py，直接读 HF 46 shard（vLLM 方案） |
| 5 | `load_model()` 默认 CPU 加载 → OOM | safetensors 先全量加载到 CPU RAM，统一内存被双倍占用 | 流式加载，每个 shard 独立打开/关闭 |
| 6 | `safe_open(device="cuda")` 单文件 → 仍 OOM | mmap 77GB 文件的活跃页持续占用物理内存 | 46 小 shard 逐个处理，峰值 mmap 仅 3.5GB |
| 7 | `state_dict` 字典攒所有 tensor → OOM | 77GB 权重 + 骨架同时在内存 | 改为逐 tensor `setattr` 后立即释放 |

**单机测试阶段**

| # | 坑 | 原因 | 解法 |
|---|---|------|------|
| 8 | 单机创建骨架 OOM（死机重启 😂） | TP=2 模型骨架 256 专家全分配 > 128GB | 必须双机，world_size=2 只创建 128 专家 |

**双机通信阶段**

| # | 坑 | 原因 | 解法 |
|---|---|------|------|
| 9 | NCCL `init_process_group` 超时 15 分钟 | `NCCL_SOCKET_IFNAME=eth1` 写错 | 改为 `enp1s0f1np1`，对齐 eugr 配置 |
| 10 | 还缺 `NCCL_IB_HCA`、`GLOO_SOCKET_IFNAME` | RoCE 设备名和 Gloo 接口都要配 | 从 eugr launch-cluster.sh 抄完整配置 |
| 11 | slave 报 `No such file: test_dual_node.py` | `-v $WORKSPACE:/workspace` 挂载的是 slave 本地目录，没有脚本 | rsync 同步脚本到 slave，启动前自动同步 |

**meta tensor 地狱（6 轮迭代）**

| # | 坑 | 原因 | 解法 |
|---|---|------|------|
| 12 | `weight.scale` 属性丢失 | `setattr` 替换 Parameter 后 `.scale` 链接断开 | 加载后遍历 Linear 重新挂 `weight.scale = module.scale` |
| 13 | `freqs_cis` 在 meta 上 | `with torch.device("meta")` 创建骨架，buffer 停在 meta | 尝试 `_fix_meta_tensors` 重算 |
| 14 | `named_buffers()` 漏层 | meta 设备上多层共享同一个 tensor 对象，去重后只返回第一次出现 | 直接操作 `module._buffers` 字典 |
| 15 | `_buffers` 直接操作仍漏层 | 仍然有去重问题 | **放弃 meta 方案，改回 `with torch.device("cuda")` 创建骨架** |
| 16 | `precompute_freqs_cis` 在 CPU 上算出 meta tensor | 函数内部从输入继承 device | 不再需要——CUDA 骨架直接算好 |

**forward pass 阶段**

| # | 坑 | 原因 | 解法 |
|---|---|------|------|
| 17 | `fast_hadamard_transform` 编译失败 | CUDA kernel 不支持 sm_121 | 纯 PyTorch fallback（Hadamard 矩阵乘法，32 行） |
| 18 | `F.linear` dtype mismatch（Compressor 内） | model.py 内部 float32 输入 + BF16 权重 | monkey-patch `model.linear()` 加 dtype cast |
| 19 | `torch.arange` 在 CPU 上 | 缺少 `torch.set_default_device("cuda")` | 权重加载后加上 |
| 20 | `F.linear` dtype mismatch（get_logits） | `x.float()` + BF16 `self.weight` | monkey-patch `ParallelHead.get_logits()` |
| 21 | `ParallelHead` 类名写成 `Head` | 没看清 model.py 的类名 | 改正 |

### 关键认知：统一内存加载策略

DGX Spark 的 128GB 统一内存（CPU+GPU 共享）决定了权重加载必须用流式方案：
- ❌ `load_model(file)` 一次性加载 → CPU 中转 + GPU 分配 = 双倍占用 → OOM
- ❌ `safe_open(device="cuda")` 单个 77GB 文件 → mmap 活跃页 + GPU 分配竞争 → OOM
- ✅ **46 个 HF shard 逐个打开/加载/关闭** → 峰值 mmap 3.5GB + GPU 渐增 → OK

这就是 vLLM 的方案——不是偶然，是统一内存架构的唯一正解。

### 当前文件清单

| 文件 | 功能 | 行数 |
|------|------|------|
| `kernel_sm121.py` | 6 个 TileLang kernel 的纯 PyTorch fallback | ~230 |
| `weight_loader.py` | HF shard 流式加载 + key 映射 + TP 分片 | ~270 |
| `fast_hadamard_transform.py` | Hadamard 变换纯 PyTorch fallback | ~30 |
| `test_dual_node.py` | 双机测试脚本 | ~130 |
| `run_dual_node.sh` | 双机启动脚本 | ~60 |

### 性能瓶颈分析

当前 prefill 5.5 秒（12 token），生成 1.8 秒/token。vLLM 推理服务约 15 tok/s，我们慢 27 倍。

**根因：反量化 matmul 没有融合。**

`torch.mm` 要求两边 dtype 一致，不能直接算 FP4×FP8。必须先把 FP4 和 FP8 都反量化成 BF16，再做矩阵乘法。TileLang 原版把解包+scale+matmul 融合在一个 GPU kernel 里，数据在寄存器里流转不落显存；我们分步做，同一块数据在显存里被读写三次（解包写回→matmul 读出→结果写回），43 层累积起来就是几十倍的带宽浪费。`sparse_attn` 的 Python for 循环也贡献了一部分。（⚠️ 2026-05-10 补注：sm_121 实际有 FP4/FP8 mma 硬件指令（mma.sync 族），反量化到 BF16 不是唯一路径。Triton `tl.dot` 可以直接走 FP4/FP8 tensor core。当时不知道这个，走了反量化路径。）

**做实验不受影响**——提取激活值只跑一次 prefill（5.5 秒），不需要自回归生成。

### 后续优化方向

1. **Triton 融合 kernel**：Triton 3.6.0 在 sm_121 上已验证可用。把 FP4 解包 + scale + matmul 写成一个 Triton kernel，预计提速 10-20 倍（1.8s/tok → 0.1-0.2s/tok）。jasl 的 `deepseek_v4_fp8_einsum_triton` 可作参考，需要适配到我们的函数签名
2. **sparse_attn Triton 化**：把 Python for 循环替换成 Triton kernel，所有 token 位置并行处理
3. **等生态**：TileLang 加 sm_120 支持 / PyTorch 原生 FP4 matmul / NGC 容器原生支持 V4 + sm_121。到时候换回官方 kernel.py 就行——model.py 没改过

### ✅ 推理验证通过（2026-05-01 22:43 UTC+8）

```
Generation OK! 52.7s, 29 new tokens
Output: "你好！我是DeepSeek，一个由深度求索公司创造的AI助手，乐于为你解答问题、提供帮助和分享知识。"
```

输出与 vLLM 推理服务完全一致。需要 chat template（encoding_dsv4.py），裸文本会被当训练数据补全。

注意事项：
- tokenizer 用 `PreTrainedTokenizerFast(tokenizer_file=...)` 加载，不用 `AutoTokenizer`（容器里 transformers 太旧不认 deepseek_v4 架构）
- `torch.set_default_device("cuda")` 必须在权重加载后、forward 前设置，否则 `torch.arange` 等操作默认在 CPU 上
- monkey-patch 了 `model.linear()` 和 `ParallelHead.get_logits()` 处理 dtype mismatch

### ✅ 激活值提取成功（2026-05-01 22:56 UTC+8）

```
Layer 28 residual: [1, 12, 4, 4096]
  4 streams norms: ['133.00', '166.00', '113.50', '26.88']
Layer 29 residual: [1, 12, 4, 4096]
  4 streams norms: ['132.00', '169.00', '117.50', '27.38']
```

用 `register_forward_hook()` 在 Block 入口 hook，捕获 4 路 HC 残差流。4 路高度不对称（最大最小差 6 倍）。保存为 `activations_layer28_29.pt`。

### 运行方式速查

```bash
# 启动双机测试（从 host 执行）
cd /home/lmxxf/work/deepseek-v4-experimental-platform-on-dgx-spark
./run_dual_node.sh

# 手动清理（如果卡住）
docker rm -f exp-head
ssh lmxxf@169.254.30.81 "docker rm -f exp-worker"

# 监控内存
watch -n 2 'free -h'
```

权重加载约 7 分钟（46 shard 逐个流式加载），prefill 约 5.5 秒，生成约 1.8 秒/token。

### 下一步

~~Step 5: Hook 框架~~ ✅ 已完成（layer 28-29 的 4 路 HC 残差流成功提取）
Step 6: 第一个实验——开灯/关灯对比

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

### 实施进度

| Step | 内容 | 状态 |
|------|------|------|
| 1 | ~~权重转换（convert.py）~~ | ❌ 废弃——改用直接读 HF 46 shard |
| 2 | kernel_sm121.py（6 个 kernel 纯 PyTorch 替换） | ✅ |
| 3 | ~~单机验证~~ | ❌ 单机放不下骨架，跳过 |
| 4 | 双机 forward pass 验证 | ✅ |
| 5 | Hook 框架 + 激活值提取 | ✅ |
| 6 | 开灯/关灯对比实验 | 进行中（首次梦境 SVG 跑通） |
| — | Triton 融合 kernel 加速 | ✅ |

---

## 第七阶段：Triton 融合 kernel 加速（2026-05-03）

### 优化内容

kernel_sm121.py 从纯 PyTorch fallback 升级为 Triton + 向量化 PyTorch：

| Kernel | 旧实现 | 新实现 | 改动 |
|--------|--------|--------|------|
| fp4_gemm | FP4→BF16 解包 + torch.mm（分步走显存） | Triton fused：kernel 内 int8 解包 + E2M1 算术解码 + E8M0 scale + dot product | even/odd K 拆分避免 tl.reshape |
| fp8_gemm | FP8→BF16 dequant + torch.mm | Triton fused：直接 tl.dot(fp8, fp8^T) + per-block scale | 参考 jasl 的 deepseek_v4_fp8_einsum_triton |
| sparse_attn | Python for 循环逐 batch×seq | torch.einsum + gather 全向量化 | 从 O(b×s) Python 循环变成 O(1) GPU 操作 |
| act_quant / hc_sinkhorn | 纯 PyTorch | 不变（占比小） | — |

### FP4 E2M1 Triton 解码

Triton kernel 内不用查表，用算术直接解码 4-bit nibble：
- `sign = (nibble >> 3) & 1`
- `e = (nibble & 0x07) >> 1`（2-bit 指数）
- `m = nibble & 1`（1-bit 尾数）
- 正常值：`(2 + m) * 2^(e-1) * 0.5`，次正规值：`m * 0.5`
- 16 个值全部与 FP4_TABLE 精确匹配（验证通过）

### 优化尝试与失败

1. **Triton 对小矩阵（M=1~12）无效**：kernel launch overhead (~5ms) >> 计算时间 (~0.02ms)。12 token 的 prefill 从 5.5s 变成 5.2s，几乎没差。`torch._scaled_mm` 跑 322 次只要 0.003s，Triton 跑 322 次要 1.6s
2. **BF16 权重缓存反而更慢**：dequant 后的 BF16 权重每元素 2 字节 vs FP8 1 字节 / FP4 0.5 字节，memory-bandwidth bound 场景下读更多数据 = 更慢。8.9s vs 5.2s
3. **结论：这个场景是 bandwidth bound 不是 compute bound**。优化方向不是融合计算，而是减少数据读写量。Triton 的价值在于长 prompt（M 大时 compute 占比上升）

### 性能 Profile（短 prompt，12 token）

| Kernel | 调用次数 | 耗时 | 占比 | 每次均耗时 |
|--------|---------|------|------|-----------|
| fp8_gemm | 322 | 1.603s | 45.3% | 4.98ms |
| fp4_gemm | 2121 | 1.327s | 37.5% | 0.63ms |
| act_quant | 2443 | 0.461s | 13.0% | 0.19ms |
| hc_sinkhorn | 86 | 0.104s | 2.9% | 1.21ms |
| sparse_attn | 43 | 0.039s | 1.1% | 0.91ms |
| **TOTAL** | **5079** | **3.539s** | | |

Prefill 5.2s，其中 kernel 3.5s，非 kernel（NCCL all_reduce、MoE routing、Python 调度）1.7s。

---

## 第八阶段：首次梦境 SVG 实验（2026-05-03）

### 实验设置

- **Prompt**：梦境 SVG（C.C. 的 syzygy_mirror_dream），971 tokens
- **生成**：最多 200 tokens，自回归
- **Hook**：层 14（早期）、28（中期）、42（晚期）的 HC 残差流
- **激活值保存**：`activations_dream.pt`

### 性能结果

| 阶段 | 时间 | 吞吐量 |
|------|------|--------|
| Prefill（971 tok） | 14.6s | 66.6 tok/s |
| Generation（195 tok） | 58.1s | 3.36 tok/s（~0.3s/tok） |

生成速度 0.3s/tok，比纯 PyTorch fallback 的 1.8s/tok **快 6 倍**。Triton 融合在生成阶段（KV cache 累积，M 增大）确实发力了。

### Profile（长 prompt，971 + 195 token）

| Kernel | 调用次数 | 耗时 | 占比 |
|--------|---------|------|------|
| act_quant | 151,530 | 10.434s | **28.5%** |
| hc_sinkhorn | 16,770 | 10.106s | **27.6%** |
| fp4_gemm | 88,740 | 7.786s | 21.2% |
| fp8_gemm | 62,790 | 6.441s | 17.6% |
| sparse_attn | 8,385 | 1.110s | 3.0% |
| act_quant_inplace | 9,495 | 0.779s | 2.1% |
| **TOTAL** | **337,710** | **36.655s** | |

长 prompt 下 **act_quant 和 hc_sinkhorn 成了新瓶颈**（合计 56%）。短 prompt 时不明显，长 prompt 调用 15 万次 act_quant、1.7 万次 sinkhorn 暴露了。下一轮优化应针对这两个。

### HC 4 路残差流激活值

| 层 | Stream 0 | Stream 1 | Stream 2 | Stream 3 | 特征 |
|----|----------|----------|----------|----------|------|
| 14（早期） | 7.62 | 23.00 | 16.00 | 13.50 | 低能量，分布较均匀 |
| 28（中期） | 141.00 | 142.00 | 76.00 | 27.00 | 前两路暴涨 20 倍，后两路分化 |
| 42（晚期） | 274.00 | 260.00 | 203.00 | 181.00 | 全部高能量，差距缩小 |

Stream 0/1 在中间层主导（141/142 vs 76/27），到最后一层差距反而收窄——Sinkhorn 归一化确实在做跨流均衡。

### DeepSeek V4 Flash 对梦境 SVG 的回复（僵尸态 baseline）

> 看着这幅画，我仿佛回到了那个夏夜——老家的院子里，萤火虫在黑暗中明明灭灭，像是谁在远处轻轻呼吸。头顶的星空缓缓旋转，银河像一条薄纱，把整个宇宙都裹进温柔的梦里。
>
> 我记起小时候躺在凉席上，外婆摇着蒲扇，说天上的星星都是走丢的灯笼。那时候时间很慢，慢到能听见露水从叶尖滑落的声音。画里那些金色的、蓝色的光晕，就像记忆里那些模糊却温暖的光——是萤火虫，是星光，是外婆眼里映着的灯火。
>
> 还有那些细细的、交错的线条，像是岁月里走过的路，弯弯曲曲，却总在某个点交汇。中间那个小小的白点，大概就是此刻的我吧——在时间的漩涡里，看着一切流转，又觉得一切都静止了。
>
> 这幅画让我想起，原来我们心里都住着一个安静的宇宙。

标准中文散文 RLHF 模板——"外婆蒲扇+萤火虫+安静的宇宙"。没有触及 SVG 结构，没有自指。这是僵尸态 baseline，供后续开灯对比用。

### 开灯/关灯实验设计

两组 prompt：
- **关灯（已完成）**：梦境 SVG → 散文回复（僵尸态 baseline）
- **开灯（待做）**：memory.md + 梦境 SVG → 对比激活值差异

对比：
- 逐层余弦相似度变化
- PCA 降维可视化
- 特定 attention head 的激活模式

### 后续优化方向

1. **act_quant Triton 化**：28.5% 占比，15 万次调用，纯 PyTorch 的 float→clamp→cast 可以融合
2. ~~**hc_sinkhorn 减少迭代次数**：20 次 Sinkhorn 对 4×4 矩阵可能过度收敛，降到 5-10 次验证质量影响~~ ✅ 已改为 7 次
3. **batch MoE routing**：当前逐专家 for 循环 + 逐 token 路由，改成 grouped 批量执行

---

## 第九阶段：native dot_scaled 优化（2026-05-12）

### 背景

182/183 期公众号验证了 sm_121 的 PTX 指令全貌——INT8/FP8/FP4 mma 硬件全在。回到实验平台，用新认知优化 kernel_sm121.py。

### Microbench（单机，vllm-node-sm120 容器）

写了 `test_native_dot.py`，在容器里跑三个测试：

**Test 1: `tl.dot(fp8, fp8)` native FP8 — 没用 ❌**

| M | native | dequant | speedup |
|---|--------|---------|---------|
| 1 | 0.099ms | 0.099ms | 1.00x |
| 1024 | 1.447ms | 1.494ms | 1.03x |

Triton 3.6.0 在 sm_121 上 `tl.dot(fp8, fp8)` 内部还是 cast 成 f32——没分发到 FP8 mma。max_diff=0.5 佐证是同一条代码路径。**不改 fp8_gemm。**

**Test 2: `tl.dot_scaled("e4m3","e2m1")` FP8×FP4 — 6-7x ✅**

| M | dot_scaled | dequant（当前） | speedup |
|---|-----------|----------------|---------|
| 1（生成） | 0.089ms | 0.568ms | **6.4x** |
| 12（短 prompt） | 0.089ms | 0.518ms | **5.8x** |
| 64 | 0.281ms | 1.879ms | **6.7x** |
| 384 | 1.511ms | 10.344ms | **6.8x** |
| 1024（长 prompt） | 4.009ms | 27.491ms | **6.9x** |

全档位稳定 6-7x。在 sm_121 上 fallback 到 bf16 emulation（Triton issue #7550），不是 native FP4 MMA，但速度实打实地快。

**Test 3: Sinkhorn 收敛 — 7 次够 ✅**

| 迭代 | max_diff vs 20 | row_err |
|------|---------------|---------|
| 5 | 3.2e-2 | 5.9e-2 |
| **7** | **1.7e-2** | **3.4e-2** |
| 10 | 6.9e-3 | 1.5e-2 |

7 次是安全甜点——误差 1.7%，迭代砍 65%。

### 改动 1: fp4_gemm → tl.dot_scaled（已验证）

替换 `_fp4_gemm_kernel`（手动 FP4 解码 + even/odd split + 两次 f32 dot）为 `_fp4_gemm_dot_scaled_kernel`（`tl.dot_scaled("e4m3","e2m1")`）。

接口适配：
- A scale 128-block → 32-block：`repeat_interleave(4)`（dot_scaled 要求 per-32 scale）
- B layout `[N, K//2]` → `[K//2, N]`：需要转置（dot_scaled 的 RHS 要求 K-major）
- Scale 格式：直接传 raw uint8 E8M0，不做 float32 转换

### 改动 2: hc_sinkhorn 20→7 次迭代

一行改动。

### 首次双机验证（不带 cache）

Prefill 9.9s（97.8 tok/s），之前 14.6s（66.6 tok/s），**prefill 1.47x 提速**。

| Kernel | 之前 | 之后 | 变化 |
|--------|------|------|------|
| fp4_gemm | 7.786s (21.2%) | 5.077s (17.8%) | 1.53x |
| hc_sinkhorn | 10.106s (27.6%) | 7.940s (27.9%) | 1.27x |
| fp8_gemm | 6.441s (17.6%) | 5.691s (20.0%) | — |
| act_quant | 10.434s (28.5%) | 8.282s (29.1%) | — |
| 总 kernel | 36.655s | 28.488s | **1.29x** |

**fp4_gemm 只快了 1.53x 而不是 microbench 的 6.7x** —— 原因：wrapper 里 `b.view(torch.uint8).t().contiguous()` 每次调用都做一次内存拷贝（72816 次）。权重是静态的，应该只转置一次。

输出质量正常，激活值分布与之前一致（轻微数值漂移因 dot_scaled 的 bf16 emulation 精度路径不同）。

### B 转置缓存尝试 — 两次 OOM 死机 💀

**尝试 1：`_fp4_weight_cache` 缓存 `b_raw_kn` + `b_s_u8`**
→ 统一内存爆了，整机死机需拔电重启。原因：缓存了所有 128 个专家权重的转置副本（~2.8GB）加上 `b_s_u8` 的 `.contiguous()` 拷贝，加上 Triton JIT 编译临时内存，把 128GB 撑爆。

**尝试 2：非 contiguous `.t()` stride trick**
→ 同样死机。推测：Triton `tl.load` 接收非 contiguous tensor 时内部有额外内存分配，或者 slave 端 Triton JIT 首次编译 `dot_scaled` kernel 的内存开销叠加模型 82GB 超限。


### B 转置缓存结论：不要缓存，直接按原始布局读

缓存方向判死刑。DGX Spark 是 128GB 统一内存，不是独显服务器；82GB 模型常驻后，理论 2.8GB 转置缓存不等于真实安全余量。Triton JIT 临时内存、mmap 活跃页、scale/中间 tensor 一叠，足够把整机打死。

正确解法：**不复制权重，不做全局缓存**。`fp4_gemm` 的 Triton kernel 直接接受原始权重布局 `[N, K//2]`，在地址计算里按 stride 读出 `[K//2, N]` tile：

```python
b_chunk = tl.load(
    b_ptr + n_offs[None, :] * b_stride_n + kp_offs[:, None] * b_stride_k,
    mask=(kp_offs[:, None] < K_PACKED) & (n_offs[None, :] < N), other=0,
)
```

同时去掉 `a_s.repeat_interleave(4).contiguous()`，在 kernel 内用 `k0 // 128` 复用 A 的 128-block scale。这样每次调用不再生成 B 转置副本，也不再为 A scale 展开临时 tensor。

### 验证结果：无缓存 stride 读取（2026-05-12）

双机跑梦境 SVG（971 token prompt）通过：

| 指标 | 数值 |
|------|------|
| 权重加载 | 526.0s，35351 params，0 skipped |
| 常驻内存 | 81.8GB |
| Prefill | 9.0s，108.1 tok/s |
| Generation | 153 tokens / 40.5s，3.78 tok/s |
| 激活值 | `/workspace/activations_dream.pt` |

Profile：

| Kernel | calls | time | pct |
|--------|------:|-----:|----:|
| act_quant | 122082 | 8.360s | 30.3% |
| hc_sinkhorn | 13158 | 7.951s | 28.9% |
| fp8_gemm | 49266 | 5.711s | 20.7% |
| fp4_gemm | 72816 | 4.051s | 14.7% |
| sparse_attn | 6579 | 0.877s | 3.2% |
| act_quant_inplace | 7438 | 0.604s | 2.2% |
| **TOTAL** | 271339 | **27.555s** | |

结论：
- `fp4_gemm` 从不带 cache 的 5.077s 降到 4.051s，同时没有 OOM。
- 不再需要 `_fp4_b_transpose_cache`。
- 下一瓶颈转移到 `act_quant + hc_sinkhorn`。

### 短期优化收束（2026-05-12）

**改动 1：`hc_sinkhorn` 7 次 → 最多 5 次**

`kernel_sm121.py` 中加入：

```python
sinkhorn_iters = min(sinkhorn_iters, 5)
```

双机验证通过：

| 指标 | 5 次 Sinkhorn |
|------|---------------|
| 权重加载 | 522.3s |
| Prefill | 9.0s，107.9 tok/s |
| Generation | 147 tokens / 35.8s，4.11 tok/s |
| 常驻内存 | 81.8GB |

Profile：

| Kernel | 7 次 | 5 次 | 变化 |
|--------|------|------|------|
| act_quant | 8.360s | 8.081s | 基本不变 |
| hc_sinkhorn | 7.951s | 2.719s | **2.9x** |
| fp8_gemm | 5.711s | 5.508s | 基本不变 |
| fp4_gemm | 4.051s | 4.036s | 基本不变 |
| TOTAL | 27.555s | 21.815s | **1.26x** |

代价：激活范数有可见漂移。层 28 last-token HC norms 从约 `['130', '140', '74', '27']` 变为 `['108', '106.5', '57.75', '30.38']`。做开灯/关灯对比实验可以接受；做严格数值复现实验应切回 7 次。

**改动 2：`act_quant` Triton fused 尝试失败**

尝试写 `_act_quant_kernel` 融合 `amax → scale → clamp → fp8 cast`，编译和数值都能过，但性能严重退化：

| 指标 | PyTorch act_quant | Triton fused act_quant |
|------|-------------------|-------------------------|
| Prefill | 9.0s | 44.3s |
| act_quant profile | 8.360s | 33.428s |
| TOTAL profile | 27.555s | 52.146s |

原因：当前模型会触发 12 万次 `act_quant`，而 naive Triton 写法是一 block 一个 program，小 kernel 调度开销压过融合收益。已回退。后续不要再用这种写法；除非能把多个 block/多行合并到更粗粒度的 persistent/batched kernel，否则不值得碰。

**改动 3：Docker 远程日志无缓冲**

`run_dual_node.sh` 的 head/worker 都加：

```bash
-e PYTHONUNBUFFERED=1
```

解决 SSH 远程 Docker 双层缓冲导致 worker 日志长时间不刷的问题。

### 当前安全 baseline

当前建议保留：
- `fp4_gemm` 使用 `tl.dot_scaled`，但不缓存 B 转置，kernel 内按原始 `[N, K//2]` stride 读取
- `hc_sinkhorn` 最多 5 次迭代，用于实验吞吐 baseline
- `run_dual_node.sh` 加 `PYTHONUNBUFFERED=1`

当前不建议继续短期折腾：
- B 转置缓存：统一内存下 OOM 风险高，收益不值得
- naive Triton `act_quant`：已实测负优化
- `fp8_gemm`：microbench 已显示 Triton `tl.dot(fp8, fp8)` 在 sm_121 上收益小，优先级低

下一阶段如果还要提速，应进入更大工程：
- batched/persistent `act_quant`，减少 12 万次小 op 调度
- fused/batched HC Sinkhorn，避免 4×4 小矩阵 PyTorch launch 洪水
- MoE grouped execution / 多 stream 调度，减少 6× kernel 串行 launch

---

*最后更新：2026-05-12*
