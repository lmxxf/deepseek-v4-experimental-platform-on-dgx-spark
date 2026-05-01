"""
双机验证脚本：TP=2 加载完整模型，跑一次 forward pass。
直接读 HF 原始 46 shard（每个 ~3.5GB），不用 convert.py 预处理。
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

# Patch Head.get_logits — it calls F.linear directly with float32 input + bf16 weight
_Head = _model_module.ParallelHead
_original_get_logits = _Head.get_logits
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

    print(f"[rank {rank}] Before init_process_group: {meminfo()}", flush=True)

    if world_size > 1:
        dist.init_process_group("nccl")

    print(f"[rank {rank}] After init_process_group: {meminfo()}", flush=True)

    def log(msg):
        print(f"[rank {rank}] {msg} | {meminfo()}", flush=True)

    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(42)

    with open(config_path) as f:
        config = json.load(f)
    config["max_batch_size"] = 1
    config["max_seq_len"] = 512
    args = ModelArgs(**config)
    log(f"Config: {args.n_layers} layers, {args.n_routed_experts} experts, dim={args.dim}")

    import model as model_module
    model_module.world_size = world_size
    model_module.rank = rank

    log("Creating model skeleton on CUDA...")
    t0 = time.time()
    with torch.device("cuda"):
        model = Transformer(args)
    t1 = time.time()
    log(f"Skeleton created in {t1-t0:.1f}s")

    from weight_loader import load_model_streaming
    log(f"Loading weights from HF shards (streaming)...")
    t2 = time.time()
    loaded, skipped = load_model_streaming(
        model, hf_ckpt_path, rank, world_size,
        n_experts=args.n_routed_experts, device="cuda", args=args
    )
    t3 = time.time()
    log(f"Weights loaded in {t3-t2:.1f}s ({loaded} params, {skipped} skipped)")

    # Check for any remaining meta tensors
    meta_params = [(n, p.shape) for n, p in model.named_parameters() if p.device.type == "meta"]
    meta_bufs = [(n, b.shape) for n, b in model.named_buffers() if b.device.type == "meta"]
    if meta_params:
        print(f"[rank {rank}] WARNING: {len(meta_params)} params still on meta!", flush=True)
        for n, s in meta_params[:5]:
            print(f"  {n}: {s}", flush=True)
    if meta_bufs:
        print(f"[rank {rank}] WARNING: {len(meta_bufs)} buffers still on meta!", flush=True)
        for n, s in meta_bufs[:5]:
            print(f"  {n}: {s}", flush=True)
    if not meta_params and not meta_bufs:
        log("All tensors on CUDA, no meta remaining")

    torch.set_default_device("cuda")

    if world_size > 1:
        dist.barrier()
    log("All ranks ready. Running forward pass...")

    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long, device="cuda")
    t4 = time.time()
    with torch.inference_mode():
        try:
            logits = model.forward(input_ids, 0)
            t5 = time.time()
            log(f"Forward pass OK! {t5-t4:.1f}s")
            log(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
            log(f"Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            top5 = logits[0].topk(5) if logits.dim() == 2 else logits.topk(5)
            log(f"Top 5 token IDs: {top5.indices.tolist()}")
        except Exception as e:
            t5 = time.time()
            log(f"Forward pass FAILED after {t5-t4:.1f}s")
            import traceback
            traceback.print_exc()

    log(f"Final GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    if world_size > 1:
        dist.destroy_process_group()
    log("Done.")


if __name__ == "__main__":
    main()
