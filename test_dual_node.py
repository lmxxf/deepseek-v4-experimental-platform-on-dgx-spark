"""
双机验证脚本：TP=2 加载完整模型，跑 prefill + 自回归生成。
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


DREAM_SVG_PATH = "/workspace/input-text.md"

def load_prompt():
    """Load the dream SVG prompt."""
    if os.path.exists(DREAM_SVG_PATH):
        with open(DREAM_SVG_PATH) as f:
            return f.read().strip()
    return "你好，请用一句话介绍你自己。"


def main():
    hf_ckpt_path = os.environ.get("HF_CKPT_PATH", "/model")
    config_path = os.environ.get("CONFIG_PATH", f"{INFERENCE_DIR}/config.json")
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "200"))

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

    # Load tokenizer and encoding
    from transformers import PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(hf_ckpt_path, "tokenizer.json"))

    encoding_dir = os.path.join(hf_ckpt_path, "encoding")
    sys.path.insert(0, encoding_dir)
    from encoding_dsv4 import encode_messages
    log("Tokenizer loaded")

    # --- Load prompt ---
    prompt_text = load_prompt()
    messages = [{"role": "user", "content": prompt_text}]
    formatted = encode_messages(messages, thinking_mode="chat")
    input_ids = torch.tensor([tokenizer.encode(formatted)], dtype=torch.long, device="cuda")
    prompt_len = input_ids.shape[1]
    log(f"Prompt: {prompt_len} tokens (dream SVG: {os.path.exists(DREAM_SVG_PATH)})")

    # === Hook: capture residual stream at layers 14, 28, 42 (early/mid/late) ===
    target_layers = [14, 28, 42]
    captured = {}

    def make_hook(layer_id):
        def hook_fn(module, input, output):
            x = input[0]
            captured[layer_id] = x.detach().cpu().clone()
        return hook_fn

    hooks = []
    for lid in target_layers:
        h = model.layers[lid].register_forward_hook(make_hook(lid))
        hooks.append(h)
    log(f"Hooks registered on layers {target_layers}")

    # === Prefill ===
    t4 = time.time()
    generated_ids = []
    with torch.inference_mode():
        try:
            logits = model.forward(input_ids, 0)
            t5 = time.time()
            log(f"Prefill OK! {t5-t4:.1f}s for {prompt_len} tokens ({prompt_len/(t5-t4):.1f} tok/s)")

            # Print captured activations
            for lid in target_layers:
                if lid in captured:
                    act = captured[lid]
                    log(f"Layer {lid} residual: shape={act.shape}")
                    log(f"  4 HC norms (last token): {[f'{act[0, -1, i].norm().item():.2f}' for i in range(min(4, act.shape[2]))]}")

            # Remove hooks after prefill (don't capture during generation)
            for h in hooks:
                h.remove()
            hooks = []

            # Save activations
            if is_main and captured:
                save_path = "/workspace/activations_dream.pt"
                torch.save({
                    "prompt": prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text,
                    "prompt_tokens": prompt_len,
                    "input_ids": input_ids.cpu(),
                    "activations": captured,
                    "target_layers": target_layers,
                    "hc_mult": 4,
                    "dim": 4096,
                }, save_path)
                log(f"Activations saved to {save_path}")

            # === Autoregressive generation ===
            log(f"Starting generation (max {max_new_tokens} tokens)...")
            next_token = logits.argmax(dim=-1)  # [1]
            generated_ids.append(next_token.item())
            start_pos = prompt_len

            eos_id = tokenizer.encode("<｜end▁of▁sentence｜>")
            eos_id = eos_id[0] if eos_id else None

            t6 = time.time()
            for step in range(max_new_tokens - 1):
                token_input = next_token.unsqueeze(0)  # [1, 1]
                logits = model.forward(token_input, start_pos)
                next_token = logits.argmax(dim=-1)
                tid = next_token.item()
                generated_ids.append(tid)
                start_pos += 1

                if eos_id is not None and tid == eos_id:
                    break

            t7 = time.time()
            gen_tokens = len(generated_ids)
            gen_time = t7 - t6
            log(f"Generation done: {gen_tokens} tokens in {gen_time:.1f}s ({gen_tokens/gen_time:.2f} tok/s)")

            # Decode and print output
            if is_main:
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                print("\n" + "="*60, flush=True)
                print("DeepSeek V4 Flash Output:", flush=True)
                print("="*60, flush=True)
                print(output_text, flush=True)
                print("="*60 + "\n", flush=True)

        except Exception as e:
            t5 = time.time()
            log(f"FAILED after {t5-t4:.1f}s")
            import traceback
            traceback.print_exc()

    # Clean up any remaining hooks
    for h in hooks:
        h.remove()

    # Print kernel profiling report
    if is_main:
        from kernel_sm121 import profile_report
        profile_report()

    log(f"Final GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    if world_size > 1:
        dist.destroy_process_group()
    log("Done.")


if __name__ == "__main__":
    main()
