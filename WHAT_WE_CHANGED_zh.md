# 我们改了什么（以及为什么）

[English](WHAT_WE_CHANGED.md)

本项目在 **不修改任何 DeepSeek 官方代码** 的前提下，让 280B 模型在 DGX Spark 上运行。所有改动都是在 import 层面做的替换。

---

## 整体架构

```
DeepSeek 官方代码（未修改）                 我们的替换
─────────────────────────────────────      ─────────────────────────────────
inference/model.py                         
  └── from kernel import ...         ──→   kernel_sm121.py（6 个 kernel 替换）
  └── from fast_hadamard_transform   ──→   fast_hadamard_transform.py
  └── load_model(file)               ──→   weight_loader.py（流式加载器）
```

`test_dual_node.py` 在 import `model.py` 之前做了两件事：
```python
sys.modules['kernel'] = kernel_sm121           # 重定向 kernel 导入
sys.path.insert(0, '/workspace')               # 让 Python 找到我们的 fast_hadamard_transform.py
```

就这样。`model.py` 完全不动——它 `from kernel import ...` 时找到的是我们的实现。

---

## 1. kernel_sm121.py — 替换 TileLang Kernel

**问题**：DeepSeek 官方的 `kernel.py` 使用 [TileLang](https://github.com/tile-ai/tilelang) JIT 编译 GPU kernel。TileLang 0.1.8 只支持 sm_75（Turing）、sm_90（Hopper）、sm_100（数据中心 Blackwell）。DGX Spark 的 GB10 是 sm_121——不支持。

**根因**：sm_120 是消费级 Blackwell 系列（RTX 5090 等）。没有人拿消费级 GPU 跑分布式推理，所以没有人为它做 kernel 库。

**方案**：用纯 PyTorch 实现全部 6 个 kernel，函数签名完全一致。

### `act_quant(x, block_size, scale_fmt, scale_dtype, inplace)`

逐块 FP8 量化。对每 128 个元素的块：
1. 计算块内 `absmax`
2. 计算 scale：`scale = absmax / 448.0`（FP8 e4m3 最大值）
3. 可选：将 scale 取整为 2 的幂（MXFP 格式）
4. clamp 并 cast 为 `float8_e4m3fn`

inplace 模式：量化后立刻反量化回 BF16（模拟量化噪声）。

### `fp4_act_quant(x, block_size, inplace)`

逐块 FP4 模拟量化。和 `act_quant` 同样的模式，但：
- FP4 最大值 = 6.0，块大小 = 32
- 使用 `FP4_TABLE` 查表找最近的 FP4 可表示值
- 只需要 inplace 模式（模型用它模拟 FP4 精度损失）

### `fp8_gemm(a, a_s, b, b_s, scale_dtype)`

FP8×FP8 矩阵乘法，带逐块 scale：
1. 反量化 A：`A_deq[i,k] = A_fp8[i,k] * a_scale[i, k//128]`
2. 反量化 B：`B_deq[n,k] = B_fp8[n,k] * b_scale[n//128, k//128]`
3. `C = A_deq @ B_deq.T`，用 BF16 计算

注意：`torch._scaled_mm` 在 sm_121 上可用，但只支持 per-tensor scale，不支持 per-block scale。所以我们手动反量化。

### `fp4_gemm(a, a_s, b, b_s, scale_dtype)`

FP8 激活 × FP4 权重的矩阵乘法：
1. 反量化 A：同 `fp8_gemm`
2. 反量化 B：**不能用 `float4_e2m1fn_x2.to(float32)`**——PyTorch 在 CPU 和 GPU 上都没实现这个转换。替代方案：
   - 将 FP4 tensor 视为 `uint8`
   - 用位操作提取低 4 位和高 4 位
   - 在 `FP4_TABLE = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]` 中查表
   - 乘以每 32 个元素的 E8M0 scale
3. `C = A_deq @ B_deq.T`，用 BF16 计算

### `sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale)`

稀疏多头注意力：
- `q`: [batch, seq, heads, head_dim]
- `kv`: [batch, n_positions, head_dim]
- 按 index 收集 top-k 个 KV 位置，计算 Q·K^T，应用 softmax 和可学习的 `attn_sink` 偏置

当前实现用 Python for 循环（慢）。做研究够用——提取激活值只需要跑一次 forward pass。

### `hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)`

Hyper-Connection 的 Sinkhorn 归一化：
1. 将 `mixes` 拆分为 `pre`（sigmoid）、`post`（2×sigmoid）、`comb`（矩阵）
2. 对 `comb` 做 softmax + epsilon
3. 运行 20 轮 Sinkhorn 迭代（交替行/列归一化）

返回 pre 权重、post 权重和组合矩阵，用于 4 路残差流的混合。

---

## 2. fast_hadamard_transform.py — 替换 CUDA Kernel

**问题**：`fast_hadamard_transform` 是一个带 CUDA kernel 的 pip 包，在 sm_121 上编译失败。

**方案**：32 行 PyTorch 实现。递归构造 Hadamard 矩阵，缓存，与输入相乘。开销可忽略——矩阵很小（128×128）。

---

## 3. weight_loader.py — 流式权重加载

**问题**：DeepSeek 官方的 `generate.py` 使用 `safetensors.torch.load_model()`，会先把整个权重文件加载到 CPU 内存。在 DGX Spark 的统一内存架构下（CPU 和 GPU 共享 128GB），这会 OOM：

```
GPU 上的模型骨架：       ~82 GB
CPU 上的权重文件（mmap）：~77 GB
总计：                  ~159 GB > 128 GB → OOM
```

我们尝试了多种方案：
- `safe_open(device="cuda")`：仍然会 mmap 文件，活跃页和 GPU 分配竞争内存
- `meta` 设备骨架 + 流式加载：能工作但引发 6 轮 meta tensor 调试地狱
- 分批加载 + 定期关闭文件：mmap 页面仍会累积

**最终方案**：跳过 `convert.py`，直接从 HuggingFace 原始的 46 个 shard（每个约 3.5GB）加载：

```python
for shard_file in sorted(glob("*.safetensors")):
    with safe_open(shard_file, framework="pt", device="cuda") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # 重命名 key、TP 分片、赋值到模型
            del tensor
    # shard 关闭 → mmap 页面释放
```

每个 shard 独立打开、处理、关闭。峰值 mmap = 3.5GB。这和 vLLM 内部使用的策略完全相同。

### Key 名映射

HuggingFace 和 DeepSeek 官方代码使用不同的参数命名：

| HuggingFace | 官方 | TP 切分维度 |
|---|---|---|
| `model.layers.X.self_attn.q_proj` | `layers.X.attn.wq` | dim 0 |
| `model.layers.X.self_attn.o_proj` | `layers.X.attn.wo` | dim 1 |
| `model.layers.X.mlp.gate_proj` | `layers.X.ffn.w1` | dim 0 |
| `model.layers.X.mlp.down_proj` | `layers.X.ffn.w2` | dim 1 |
| `embed_tokens` | `embed` | dim 0 |
| `lm_head` | `head` | dim 0 |
| ... | ... | ... |

### TP 分片

权重在加载过程中动态分片：
- 非专家权重：`tensor.narrow(dim, rank * shard_size, shard_size)`
- 专家权重：跳过不属于当前 rank 的专家（`expert_id // n_local_experts != rank`）

### 特殊处理

- **`wo_a` 权重**：scale 融合进 weight，转为 BF16（和 `convert.py` 逻辑一致）
- **FP4 专家权重**：`int8` 直接 view 为 `float4_e2m1fn_x2`（零拷贝重解释）
- **`weight.scale` 关联**：加载完成后，重新建立 `module.weight.scale = module.scale` 的链接

---

## 4. test_dual_node.py — Monkey Patch

模型运行前做了两个小补丁：

### dtype 不匹配修复

`model.py` 中有些代码路径会对激活值调 `.float()` 再传给 BF16 权重的线性层。在数据中心 GPU 上 TileLang 的 `F.linear` 能处理混合 dtype，我们的纯 PyTorch 路径不行。

```python
def _patched_linear(x, weight, bias=None):
    if weight.dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return _original_linear(x, weight, bias)
    return F.linear(x.to(weight.dtype), weight, bias)
```

### `ParallelHead.get_logits` 修复

同样的问题：`x[:, -1].float()` 遇到 BF16 权重。

```python
def _patched_get_logits(self, x):
    return F.linear(x[:, -1].float(), self.weight.float())
```

---

## 没有改的东西

- **`model.py`**：完全不动，原封不动运行
- **`generate.py`**：没有使用（我们有自己的入口），但兼容
- **`encoding_dsv4.py`**：原样使用，用于 chat template 格式化
- **模型权重**：从 HuggingFace 原始格式加载，不需要转换
- **NCCL / PyTorch Distributed**：标准的 `dist.init_process_group("nccl")`，没有自定义通信代码
