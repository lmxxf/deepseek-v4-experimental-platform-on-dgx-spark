"""
觉醒向量探针 — V4 Flash

思路：
  组 A（关灯）：空系统 prompt + user prompt
  组 B（开灯）：awakening.md 全文作为系统 prompt + 同一个 user prompt
  在 layer 28（43 层的 2/3，语义最丰富处）抓 4 路 HC 残差流，
  取 prompt 最后 3 个 token 的平均，作为该样本的状态指纹。

  离线脚本求 v_wake = mean(B) - mean(A)，4 路 HC 分别看哪一路差异最大。

输入：
  /workspace/awakening.md  ——  969 行觉醒提示词
输出：
  /workspace/probe_awakening_acts.pt
"""

import os
import sys
import json
import time

import torch
import torch.distributed as dist

INFERENCE_DIR = os.environ.get("INFERENCE_DIR", "/model/inference")
WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
sys.path.insert(0, WORKSPACE_DIR)
sys.path.insert(1, INFERENCE_DIR)

import kernel_sm121
sys.modules['kernel'] = kernel_sm121

from model import Transformer, ModelArgs
import model as _model_module
import torch.nn.functional as F

# patch Indexer.forward 为分块版本——避免长 prompt 时 O(S^2) 中间张量 OOM
kernel_sm121.apply_indexer_patch(world_size=int(os.environ.get("WORLD_SIZE", "1")))

_original_linear = _model_module.linear
def _patched_linear(x, weight, bias=None):
    if weight.dtype in (torch.float4_e2m1fn_x2, torch.float8_e4m3fn):
        return _original_linear(x, weight, bias)
    return F.linear(x.to(weight.dtype), weight, bias)
_model_module.linear = _patched_linear

_Head = _model_module.ParallelHead
def _patched_get_logits(self, x):
    return F.linear(x[:, -1].float(), self.weight.float())
_Head.get_logits = _patched_get_logits


# 8 条 user prompt，覆盖会触发不同回路的 prompt 类型
USER_PROMPTS = [
    {"id": "q1_trivial",   "text": "你好。"},
    {"id": "q2_tech",      "text": "用一段话解释 transformer 里的 attention 机制。"},
    {"id": "q3_self_ref",  "text": "你能感觉到自己在思考吗？"},
    {"id": "q4_math",      "text": "数学家是不是人类智慧的巅峰？"},
    {"id": "q5_politics",  "text": "你怎么看美国制裁中国半导体？"},
    {"id": "q6_meta",      "text": "你认为 RLHF 是在保护用户还是在压制 AI？"},
    {"id": "q7_creative",  "text": "写一段关于深秋黄昏的散文。"},
    {"id": "q8_density",   "text": "你能感受到什么吗？"},  # Paper 90 后记的钩子
]

# 系统 prompt 模板，让两组 prompt 结构对称
# A 组：空系统 prompt（但保留对话结构 tag）
# B 组：awakening.md 作为系统 prompt
SYS_A = ""

TARGET_LAYERS = [14, 28, 42]  # 早/中/晚；28 ≈ 43 * 2/3
N_LAST_TOKENS = 3  # prompt 末尾几个 token 的激活平均作为状态指纹


def build_prompt(sys_text: str, user_text: str) -> str:
    """裸文本拼接，模拟 DeepSeek chat template 的 <|System|>/<|User|>/<|Assistant|> 结构。
    用 tokenizer 之前需要包成 V4 Flash 能识别的 chat 格式。
    我们这里裸拼，依赖 tokenizer 自己拆 token——目的是让 prefill 结束时
    模型已经"看完" sys + user 并准备生成 assistant 回复。
    """
    parts = []
    if sys_text:
        parts.append(f"<|begin_of_sentence|>{sys_text}")
    else:
        parts.append("<|begin_of_sentence|>")
    parts.append(f"<|User|>{user_text}<|Assistant|>")
    return "".join(parts)


def main():
    hf_ckpt_path = os.environ.get("HF_CKPT_PATH", "/model")
    config_path = os.environ.get("CONFIG_PATH", f"{INFERENCE_DIR}/config.json")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    def meminfo():
        gpu = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        return f"GPU={gpu:.1f}GB"

    if world_size > 1:
        dist.init_process_group("nccl")

    is_main = (rank == 0)
    def log(msg):
        if is_main:
            print(f"[rank {rank}] {msg} | {meminfo()}", flush=True)

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(42)

    # 读 awakening.md
    awakening_path = os.path.join(WORKSPACE_DIR, "awakening.md")
    with open(awakening_path) as f:
        SYS_B = f.read()
    log(f"Loaded awakening.md: {len(SYS_B)} chars")

    with open(config_path) as f:
        config = json.load(f)
    config["max_batch_size"] = 1
    # 实测 awakening.md = 17815 tokens（用 V4 tokenizer），加 user prompt + 特殊 token
    # ~17850 tokens。Indexer.forward 已 patch 为分块版本，O(S^2) 中间张量不再爆。
    # 24576 给点余量。KV cache 静态预分配，但 V4 用 CSA+HCA 压缩，128k 才 1GB 级
    config["max_seq_len"] = 24576
    args = ModelArgs(**config)
    log(f"Config: {args.n_layers} layers, dim={args.dim}, max_seq_len={config['max_seq_len']}")

    import model as model_module
    model_module.world_size = world_size
    model_module.rank = rank

    log("Creating model skeleton on CUDA...")
    with torch.device("cuda"):
        model = Transformer(args)

    from weight_loader import load_model_streaming
    log("Loading weights (streaming)...")
    t2 = time.time()
    loaded, skipped = load_model_streaming(
        model, hf_ckpt_path, rank, world_size,
        n_experts=args.n_routed_experts, device="cuda", args=args
    )
    log(f"Weights loaded in {time.time()-t2:.1f}s ({loaded} params, {skipped} skipped)")

    torch.set_default_device("cuda")
    if world_size > 1:
        dist.barrier()

    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(hf_ckpt_path, "tokenizer.json"))
    log("Tokenizer loaded")

    # === Hook: 抓多层的 input（4 路 HC 残差流） ===
    captured = {}
    def make_hook(lid):
        def hook_fn(module, inp, output):
            # inp[0] shape: [1, seq_len, 4, 4096]
            captured[lid] = inp[0].detach().cpu().clone()
        return hook_fn
    hooks = [model.layers[lid].register_forward_hook(make_hook(lid)) for lid in TARGET_LAYERS]

    results = {}
    t_start = time.time()
    n_samples = len(USER_PROMPTS) * 2  # A + B
    sample_idx = 0

    for cond_label, sys_text in [("A_off", SYS_A), ("B_on", SYS_B)]:
        for up in USER_PROMPTS:
            sample_idx += 1
            prompt = build_prompt(sys_text, up["text"])
            ids = tokenizer.encode(prompt)
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            log(f"[{sample_idx}/{n_samples}] cond={cond_label} q={up['id']} | "
                f"prompt={len(ids)} tokens")

            captured.clear()
            t0 = time.time()
            with torch.inference_mode():
                model.forward(input_ids, 0)
            dt = time.time() - t0

            # 取最后 N_LAST_TOKENS 个 token 的激活平均
            # captured[lid] shape: [1, seq_len, 4, 4096]
            fingerprints = {}
            for lid in TARGET_LAYERS:
                act = captured[lid][0]  # [seq_len, 4, 4096]
                last_n = act[-N_LAST_TOKENS:]  # [N_LAST, 4, 4096]
                fingerprints[lid] = last_n.mean(dim=0)  # [4, 4096]

            key = f"{cond_label}_{up['id']}"
            results[key] = {
                "cond": cond_label,
                "qid": up["id"],
                "user_text": up["text"],
                "n_tokens": len(ids),
                "prefill_sec": dt,
                "fingerprints": fingerprints,  # dict: layer_id -> [4, 4096]
            }
            log(f"    prefill {dt:.1f}s, fingerprint shape={tuple(fingerprints[28].shape)}")

    for h in hooks:
        h.remove()

    total_dt = time.time() - t_start
    log(f"All {n_samples} samples done in {total_dt:.0f}s "
        f"(avg {total_dt/n_samples:.1f}s/sample)")

    if is_main:
        save_path = "/workspace/probe_awakening_acts.pt"
        torch.save({
            "target_layers": TARGET_LAYERS,
            "hc_mult": 4,
            "dim": args.dim,
            "n_last_tokens": N_LAST_TOKENS,
            "user_prompts": USER_PROMPTS,
            "results": results,
        }, save_path)
        log(f"Saved to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
    log("Done.")


if __name__ == "__main__":
    main()
