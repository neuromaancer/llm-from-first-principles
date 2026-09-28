"""Neural-network building blocks used by the decoder-only Transformer."""

from .attention import (
    CausalSelfAttention,
    GroupedQueryAttention,
    expand_kv_heads,
)
from .fast_weights import (
    additive_memory_update,
    delta_memory_update,
    read_fast_weight_memory,
)
from .linear_attention import (
    CausalLinearAttention,
    LinearAttentionState,
    positive_feature_map,
)
from .mlp import SwiGLU
from .rope import apply_rope, build_rope_cos_sin, rope_frequencies
from .transformer import TransformerBlock

__all__ = [
    "CausalLinearAttention",
    "CausalSelfAttention",
    "GroupedQueryAttention",
    "LinearAttentionState",
    "SwiGLU",
    "TransformerBlock",
    "additive_memory_update",
    "apply_rope",
    "build_rope_cos_sin",
    "delta_memory_update",
    "expand_kv_heads",
    "positive_feature_map",
    "read_fast_weight_memory",
    "rope_frequencies",
]
