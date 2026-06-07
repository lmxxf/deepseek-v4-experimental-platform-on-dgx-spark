"""
诗歌虫洞实证 — V4 Flash 探针阶段
一次加载，循环跑「诗 + 白话」prompt（裸文本，不套 chat template）。
抽 layer 42 的 4 路 HC 残差流 [1, seq_len, 4, 4096]，原样保存。
折算方式（stream0/求和/拼接）留给离线分析脚本决定。
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

# Monkey-patch linear() to handle dtype mismatch (float32 input + bfloat16 weight)
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


# 探针阶段：只跑 1 组，省 prefill 时间。验证信号后再上全部 5 组。
PAIRS = [
    {
        "id": "3_sensory_conflict",
        "type": "感知通道冲突",
        "poem": "空山不见人，但闻人语响。",
        "plain": "山里空荡荡的看不见人，只是偶尔听到有人说话的声音。",
        "bridge_token": "闻",
        "source": "王维《鹿柴》",
    },
]

TARGET_LAYERS = [14, 28, 42]  # 早/中/晚；28 ≈ 43 层的 2/3，语义信息最丰富处


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

    with open(config_path) as f:
        config = json.load(f)
    config["max_batch_size"] = 1
    config["max_seq_len"] = 4096
    args = ModelArgs(**config)
    log(f"Config: {args.n_layers} layers, dim={args.dim}")

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

    # tokenizer — 裸文本编码，不套 chat template
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(hf_ckpt_path, "tokenizer.json"))
    log("Tokenizer loaded")

    # === Hook: 抓多层的输入（4 路 HC 残差流） ===
    captured = {}
    def make_hook(lid):
        def hook_fn(module, inp, output):
            captured[lid] = inp[0].detach().cpu().clone()
        return hook_fn
    hooks = [model.layers[lid].register_forward_hook(make_hook(lid)) for lid in TARGET_LAYERS]

    results = {}
    for pair in PAIRS:
        for label in ("poem", "plain"):
            text = pair[label]
            ids = tokenizer.encode(text)
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            tokens = [tokenizer.decode([t]) for t in ids]
            log(f"[{pair['id']}/{label}] '{text}' -> {len(ids)} tokens: {tokens}")

            captured.clear()
            with torch.inference_mode():
                model.forward(input_ids, 0)

            acts = {lid: captured[lid] for lid in TARGET_LAYERS}  # 每层 [1, seq_len, 4, 4096]
            log(f"  captured layers={list(acts.keys())}, shape={tuple(acts[TARGET_LAYERS[0]].shape)}")
            results[f"{pair['id']}_{label}"] = {
                "text": text,
                "tokens": tokens,
                "input_ids": input_ids.cpu(),
                "activations": acts,
                "bridge_token": pair["bridge_token"],
                "type": pair["type"],
                "source": pair["source"],
            }

    for h in hooks:
        h.remove()

    if is_main:
        save_path = "/workspace/probe_poem_acts.pt"
        torch.save({
            "target_layers": TARGET_LAYERS,
            "hc_mult": 4,
            "dim": args.dim,
            "results": results,
        }, save_path)
        log(f"Saved probe activations to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
    log("Done.")


if __name__ == "__main__":
    main()
