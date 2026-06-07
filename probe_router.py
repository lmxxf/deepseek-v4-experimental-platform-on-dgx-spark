"""
MoE 路由探针 — 看诗里每个字分别点亮哪些专家。
V4 Flash: 256 专家选 6 + 1 共享。前 3 层哈希路由(跟语义无关)，第 3 层起分数路由。
hook 每层 Gate 输出的 indices [n_tokens, 6]，存下来离线分析。
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

from model import Transformer, ModelArgs, Gate
import model as _model_module
import torch.nn.functional as F

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


def main():
    hf_ckpt_path = os.environ.get("HF_CKPT_PATH", "/model")
    config_path = os.environ.get("CONFIG_PATH", f"{INFERENCE_DIR}/config.json")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group("nccl")
    is_main = (rank == 0)
    def log(msg):
        if is_main:
            gpu = torch.cuda.memory_allocated() / 1e9
            print(f"[rank {rank}] {msg} | GPU={gpu:.1f}GB", flush=True)

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(42)

    with open(config_path) as f:
        config = json.load(f)
    config["max_batch_size"] = 1
    config["max_seq_len"] = 4096
    args = ModelArgs(**config)
    n_hash = args.n_hash_layers
    log(f"Config: {args.n_layers} layers, {args.n_routed_experts} experts, "
        f"top-{args.n_activated_experts}, n_hash_layers={n_hash}")

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

    # === Hook 所有 Gate，抓每层每个 token 的 top-k 专家 id ===
    # Gate.forward 返回 (weights, indices)，indices: [n_tokens, topk]
    captured = {}  # layer_id -> indices tensor
    gate_layer_map = {}
    for lid, layer in enumerate(model.layers):
        gate = layer.ffn.gate
        gate_layer_map[id(gate)] = lid

    def gate_hook(module, inp, output):
        weights, indices = output
        lid = gate_layer_map[id(module)]
        captured[lid] = indices.detach().cpu().clone()

    hooks = [layer.ffn.gate.register_forward_hook(gate_hook) for layer in model.layers]
    log(f"Registered {len(hooks)} gate hooks")

    results = {}
    for pair in PAIRS:
        for label in ("poem", "plain"):
            text = pair[label]
            ids = tokenizer.encode(text)
            input_ids = torch.tensor([ids], dtype=torch.long, device="cuda")
            tokens = [tokenizer.decode([t]) for t in ids]
            log(f"[{pair['id']}/{label}] '{text}' -> {len(ids)} tokens")

            captured.clear()
            with torch.inference_mode():
                model.forward(input_ids, 0)

            # captured[lid]: [n_tokens, topk]
            routing = {lid: captured[lid] for lid in sorted(captured)}
            log(f"  captured {len(routing)} layers, indices shape={tuple(routing[0].shape)}")
            results[f"{pair['id']}_{label}"] = {
                "text": text,
                "tokens": tokens,
                "input_ids": ids,
                "routing": routing,
                "bridge_token": pair["bridge_token"],
                "type": pair["type"],
                "source": pair["source"],
            }

    for h in hooks:
        h.remove()

    if is_main:
        save_path = "/workspace/probe_router.pt"
        torch.save({
            "n_layers": args.n_layers,
            "n_routed_experts": args.n_routed_experts,
            "topk": args.n_activated_experts,
            "n_hash_layers": n_hash,
            "results": results,
        }, save_path)
        log(f"Saved routing to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
    log("Done.")


if __name__ == "__main__":
    main()
