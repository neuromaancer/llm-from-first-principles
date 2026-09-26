"""Type aliases for per-layer and model-level KV caches."""

import torch

LayerKVCache = tuple[
    torch.Tensor,
    torch.Tensor,
]

KVCache = list[LayerKVCache]

__all__ = [
    "KVCache",
    "LayerKVCache",
]
