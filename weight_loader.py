"""
weight_loader.py — Stream HF shard weights directly to GPU with key mapping + TP sharding.

Replaces convert.py's offline conversion. Reads 46 small HF shards (~3.5GB each)
one at a time, does key renaming + TP slicing on the fly, yields (name, tensor) pairs.

Peak memory = final model size + one shard (~3.5GB), no 77GB mmap.
"""

import os
from glob import glob
from typing import Generator, Tuple

import torch
from safetensors import safe_open


# HF name → official name, TP split dimension (None = no split, replicate)
KEY_MAPPING = {
    "embed_tokens": ("embed", 0),
    "input_layernorm": ("attn_norm", None),
    "post_attention_layernorm": ("ffn_norm", None),
    "q_proj": ("wq", 0),
    "q_a_proj": ("wq_a", None),
    "q_a_layernorm": ("q_norm", None),
    "q_b_proj": ("wq_b", 0),
    "kv_a_proj_with_mqa": ("wkv_a", None),
    "kv_a_layernorm": ("kv_norm", None),
    "kv_b_proj": ("wkv_b", 0),
    "o_proj": ("wo", 1),
    "gate_proj": ("w1", 0),
    "down_proj": ("w2", 1),
    "up_proj": ("w3", 0),
    "lm_head": ("head", 0),
    "embed": ("embed", 0),
    "wq_b": ("wq_b", 0),
    "wo_a": ("wo_a", 0),
    "wo_b": ("wo_b", 1),
    "head": ("head", 0),
    "attn_sink": ("attn_sink", 0),
    "weights_proj": ("weights_proj", 0),
}


def _rename_key(name: str) -> Tuple[str, int | None]:
    """Rename HF key to official key and return TP split dimension."""
    if name.startswith("model."):
        name = name[len("model."):]
    name = name.replace("self_attn", "attn")
    name = name.replace("mlp", "ffn")
    name = name.replace("weight_scale_inv", "scale")
    name = name.replace("e_score_correction_bias", "bias")

    if any(x in name for x in ["hc", "attn_sink", "tie2eid", "ape"]):
        key = name.split(".")[-1]
    else:
        key = name.split(".")[-2]

    if key in KEY_MAPPING:
        new_key, dim = KEY_MAPPING[key]
    else:
        new_key, dim = key, None

    name = name.replace(key, new_key)
    return name, dim


def stream_weights(
    hf_ckpt_path: str,
    rank: int,
    world_size: int,
    n_experts: int = 256,
    device: str = "cuda",
) -> Generator[Tuple[str, torch.Tensor], None, None]:
    """Stream weights from HF shards, one tensor at a time.

    Yields (official_name, tensor_on_device) pairs.
    Each shard is opened/closed independently — peak mmap = one shard (~3.5GB).
    """
    n_local_experts = n_experts // world_size
    expert_start = rank * n_local_experts
    expert_end = expert_start + n_local_experts

    shard_files = sorted(glob(os.path.join(hf_ckpt_path, "*.safetensors")))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files in {hf_ckpt_path}")

    # Collect wo_a weight+scale pairs (need special handling: fuse scale into weight)
    # We buffer these per-shard since weight and scale are in the same shard
    wo_a_buffer = {}

    for shard_idx, shard_file in enumerate(shard_files):
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for orig_name in f.keys():
                # Skip MTP embedding/head duplicates
                if orig_name.startswith("model.mtp.") and ("emb" in orig_name or orig_name.endswith("head.weight")):
                    continue

                param = f.get_tensor(orig_name)
                name, dim = _rename_key(orig_name)

                # Expert routing: skip experts not belonging to this rank
                if "experts" in name and "shared_experts" not in name:
                    idx = int(name.split(".")[-3])
                    if idx < expert_start or idx >= expert_end:
                        del param
                        continue

                # TP sharding
                if dim is not None and world_size > 1:
                    shard_size = param.size(dim) // world_size
                    param = param.narrow(dim, rank * shard_size, shard_size).contiguous()

                # wo_a special handling: fuse scale into weight, convert to BF16
                if name.endswith("wo_a.weight"):
                    wo_a_buffer[name] = param
                    del param
                    continue
                elif name.endswith("wo_a.scale"):
                    weight_name = name.replace("scale", "weight")
                    if weight_name in wo_a_buffer:
                        weight = wo_a_buffer.pop(weight_name)
                        weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * param[:, None, :, None].float()
                        weight = weight.flatten(2, 3).flatten(0, 1).bfloat16()
                        yield weight_name, weight.to(device)
                        del weight, param
                        continue
                    else:
                        wo_a_buffer[name] = param
                        del param
                        continue

                # FP4 expert weights: int8 → view as float4_e2m1fn_x2
                if "experts" in name and param.dtype == torch.int8:
                    param = param.view(torch.float4_e2m1fn_x2)

                yield name, param.to(device)
                del param

        # Flush any remaining wo_a pairs from this shard
        wo_a_names = list(wo_a_buffer.keys())
        for buf_name in wo_a_names:
            if buf_name.endswith("wo_a.weight"):
                scale_name = buf_name.replace("weight", "scale")
                if scale_name in wo_a_buffer:
                    weight = wo_a_buffer.pop(buf_name)
                    scale = wo_a_buffer.pop(scale_name)
                    weight = weight.unflatten(0, (-1, 128)).unflatten(-1, (-1, 128)).float() * scale[:, None, :, None].float()
                    weight = weight.flatten(2, 3).flatten(0, 1).bfloat16()
                    yield buf_name, weight.to(device)
                    del weight, scale

        print(f"  shard {shard_idx+1}/{len(shard_files)} done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Yield any remaining buffered wo_a tensors
    for buf_name, buf_tensor in wo_a_buffer.items():
        yield buf_name, buf_tensor.to(device)


def load_model_streaming(
    model: torch.nn.Module,
    hf_ckpt_path: str,
    rank: int,
    world_size: int,
    n_experts: int = 256,
    device: str = "cuda",
    args = None,
) -> Tuple[int, int]:
    """Load weights into a meta-device model by streaming from HF shards.

    Args:
        args: ModelArgs, needed to recompute freqs_cis (RoPE frequencies).

    Returns (loaded_count, skipped_count).
    """
    loaded, skipped = 0, 0

    for name, tensor in stream_weights(hf_ckpt_path, rank, world_size, n_experts, device):
        parts = name.split(".")
        obj = model
        try:
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            attr_name = parts[-1]
            if hasattr(obj, attr_name):
                current = getattr(obj, attr_name)
                if isinstance(current, torch.nn.Parameter):
                    setattr(obj, attr_name, torch.nn.Parameter(tensor, requires_grad=False))
                else:
                    setattr(obj, attr_name, tensor)
                loaded += 1
            else:
                skipped += 1
        except (AttributeError, IndexError, TypeError):
            skipped += 1
        del tensor

    # Re-link weight.scale attributes (model.py's Linear stores self.weight.scale = self.scale)
    from model import Linear
    for module in model.modules():
        if isinstance(module, Linear) and module.scale is not None:
            module.weight.scale = module.scale

    _fix_meta_tensors(model, args, device)

    return loaded, skipped


def _fix_meta_tensors(model, args, device):
    """Move all remaining meta-device tensors to CUDA and recompute freqs_cis."""
    from model import precompute_freqs_cis

    # Fix ALL meta tensors. named_buffers() deduplicates shared meta objects,
    # so we must iterate every module directly.

    def _recompute_freqs(rope_head_dim, compress_ratio):
        if compress_ratio:
            original_seq_len = args.original_seq_len
            rope_theta = args.compress_rope_theta
        else:
            original_seq_len = 0
            rope_theta = args.rope_theta
        # Compute on CPU then move to device — avoids meta device context leaking
        freqs = precompute_freqs_cis(
            rope_head_dim, args.max_seq_len, original_seq_len,
            rope_theta, args.rope_factor, args.beta_fast, args.beta_slow,
        )
        return freqs.to(device) if freqs.device.type != device.split(":")[0] else freqs

    # 1. Fix every module's _buffers directly (bypass named_buffers dedup)
    for module in model.modules():
        for buf_name in list(module._buffers.keys()):
            buf = module._buffers[buf_name]
            if buf is None or buf.device.type != "meta":
                continue
            if buf_name == "freqs_cis":
                rope_dim = getattr(module, 'rope_head_dim', args.rope_head_dim)
                cr = getattr(module, 'compress_ratio', 0)
                module._buffers[buf_name] = _recompute_freqs(rope_dim, cr)
                print(f"  Fixed freqs_cis for {module.__class__.__name__}: {module._buffers[buf_name].shape} on {module._buffers[buf_name].device}", flush=True)
            elif "score_state" in buf_name:
                module._buffers[buf_name] = torch.full(buf.shape, float("-inf"), dtype=buf.dtype, device=device)
            else:
                module._buffers[buf_name] = torch.zeros(buf.shape, dtype=buf.dtype, device=device)

    # 2. Fix non-buffer, non-parameter meta tensors in module vars
    for module in model.modules():
        for attr_name in list(vars(module).keys()):
            if attr_name.startswith('_'):
                continue
            val = getattr(module, attr_name, None)
            if isinstance(val, torch.Tensor) and not isinstance(val, torch.nn.Parameter) and val.device.type == "meta":
                setattr(module, attr_name, torch.zeros(val.shape, dtype=val.dtype, device=device))

    # 3. Final sweep: verify nothing is still on meta
    remaining = []
    for name, buf in model.named_buffers():
        if buf.device.type == "meta":
            remaining.append(name)
    for module in model.modules():
        for attr_name in list(vars(module).keys()):
            val = getattr(module, attr_name, None)
            if isinstance(val, torch.Tensor) and val.device.type == "meta":
                remaining.append(f"{module.__class__.__name__}.{attr_name}")
    if remaining:
        print(f"  WARNING: still {len(remaining)} meta tensors after fix: {remaining[:5]}", flush=True)
