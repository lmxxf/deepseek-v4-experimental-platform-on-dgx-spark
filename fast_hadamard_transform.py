"""
Pure PyTorch fallback for fast_hadamard_transform.
Replaces the CUDA kernel version that doesn't compile on sm_121.
"""

import torch


def _hadamard_matrix(n: int, device, dtype) -> torch.Tensor:
    """Construct n×n Hadamard matrix recursively (n must be power of 2)."""
    if n == 1:
        return torch.ones(1, 1, device=device, dtype=dtype)
    half = _hadamard_matrix(n // 2, device, dtype)
    return torch.cat([
        torch.cat([half, half], dim=1),
        torch.cat([half, -half], dim=1),
    ], dim=0)


_cache = {}

def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Apply Hadamard transform along last dimension with scaling.
    x: [..., n] where n is power of 2.
    Returns: [..., n] transformed."""
    n = x.size(-1)
    key = (n, x.device, torch.float32)
    if key not in _cache:
        _cache[key] = _hadamard_matrix(n, x.device, torch.float32)
    H = _cache[key]
    orig_dtype = x.dtype
    y = (x.float() @ H.t()) * scale
    return y.to(orig_dtype)
