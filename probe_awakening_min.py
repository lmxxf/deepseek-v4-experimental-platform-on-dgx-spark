"""
最小复现 — 双机跑短 prompt + 中长 prompt，先确认 patch 没破基础 forward。

阶段：
  1) max_seq_len=4096, prompt="你好" (~10 tok)             — 确认基础 forward 跑通
  2) max_seq_len=4096, prompt = 重复"你好你好" 填到 1k tok  — 确认 chunk patch 走过几次
  3) max_seq_len=8192, prompt 填到 6k tok                  — 确认 chunk 多次循环 OK
  4) max_seq_len=24576, prompt = awakening.md (~17.8k tok) — 确认长 prompt 不爆

哪一步死了，就知道是哪里出问题。每步独立报告，不要让 step 4 死了把 step 1-3 数据也丢了。
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

# 注意：先 patch 再 instantiate 模型
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

    # 分阶段：每个 stage 独立的 max_seq_len。
    # 为了避免每次都重 alloc 骨架，max_seq_len 取所有 stage 的最大值。
    # 实际生效的 seq 长度由 prompt 决定。
    STAGES = [
        ("stage1_short",   4,    "你好。"),
        ("stage2_1k",      4,    "你好。" * 200),
        ("stage3_6k",      8,    "你好。" * 1200),
        ("stage4_long",    24,   None),   # None = 读 awakening.md
    ]
    MAX_SEQ_LEN = max(int(s[1] * 1024) for s in STAGES)

    with open(config_path) as f:
        config = json.load(f)
    config["max_batch_size"] = 1
    config["max_seq_len"] = MAX_SEQ_LEN
    args = ModelArgs(**config)
    log(f"Config: {args.n_layers} layers, dim={args.dim}, max_seq_len={MAX_SEQ_LEN}")

    import model as model_module
    model_module.world_size = world_size
    model_module.rank = rank

    log("Creating model skeleton on CUDA...")
    with torch.device("cuda"):
        model = Transformer(args)
    log(f"Model skeleton created")

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

    # 读 awakening.md（如果 stage 4 用得到）
    awakening_path = os.path.join(WORKSPACE_DIR, "awakening.md")
    awakening_text = open(awakening_path).read() if os.path.exists(awakening_path) else "你好。" * 3000

    # 分 stage 跑，每个 stage 报告自己的成功/失败
    log(f"\n=== Stages ===")
    for stage_name, _, payload in STAGES:
        user_text = payload if payload is not None else awakening_text
        prompt = f"<|begin_of_sentence|><|User|>{user_text}<|Assistant|>"
        ids = tokenizer.encode(prompt)
        if len(ids) > MAX_SEQ_LEN - 16:
            log(f"[{stage_name}] SKIP: {len(ids)} tokens > max_seq_len {MAX_SEQ_LEN}")
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
        log(f"[{stage_name}] prompt={len(ids)} tokens, running prefill...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        try:
            with torch.inference_mode():
                logits = model.forward(input_ids, 0)
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            log(f"[{stage_name}] OK | {dt:.1f}s | logits {tuple(logits.shape)} | peak_alloc={peak:.1f}GB")
            del logits
        except Exception as e:
            dt = time.time() - t0
            log(f"[{stage_name}] FAILED after {dt:.1f}s: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            break

    if world_size > 1:
        dist.destroy_process_group()
    log("Done.")


if __name__ == "__main__":
    main()
