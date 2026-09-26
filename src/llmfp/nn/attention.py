"""Causal attention variants used by the decoder-only Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .rope import apply_rope, build_rope_cos_sin


def expand_kv_heads(
    x: torch.Tensor,
    num_query_heads: int,
) -> torch.Tensor:
    """Expand compact KV heads to align with query heads.

    Each KV head is repeated for every query head assigned to its group.

    Args:
        x: Key or value tensor with shape (B, H_KV, T, D).
        num_query_heads: Number of query heads H_Q.

    Returns:
        Expanded tensor with shape (B, H_Q, T, D).

    Raises:
        ValueError: If H_Q is not divisible by H_KV.
    """
    if x.ndim != 4:
        raise ValueError(
            "x must have shape (B, H_KV, T, D)."
        )

    num_kv_heads: int = x.shape[1]

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads."
        )

    queries_per_kv_head: int = (
        num_query_heads // num_kv_heads
    )

    # Example: [K0, K1] with repeats=2 becomes
    # [K0, K0, K1, K1].
    return torch.repeat_interleave(
        x,
        repeats=queries_per_kv_head,
        dim=1,
    )


class GroupedQueryAttention(nn.Module):
    """Causal attention supporting MHA, GQA, and MQA with KV caching.

    The number of query heads and key/value heads are configured
    independently.

    Boundary cases:
        - H_KV == H_Q: Multi-Head Attention (MHA)
        - 1 < H_KV < H_Q: Grouped-Query Attention (GQA)
        - H_KV == 1: Multi-Query Attention (MQA)

    Args:
        embedding_dim: Width of the residual stream.
        num_query_heads: Number of independent query heads.
        num_kv_heads: Number of key/value heads.
        rope_base: Base controlling RoPE frequencies.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_query_heads: int,
        num_kv_heads: int,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        if embedding_dim % num_query_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_query_heads."
            )

        if num_query_heads % num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads."
            )

        self.embedding_dim: int = embedding_dim
        self.num_query_heads: int = num_query_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = (
            embedding_dim // num_query_heads
        )
        self.rope_base: float = rope_base

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE."
            )

        # Q keeps the full query-head width H_Q * D = C.
        self.query_projection = nn.Linear(
            embedding_dim,
            num_query_heads * self.head_dim,
            bias=False,
        )

        # K/V only produce H_KV heads, reducing projection width and
        # the persistent KV-cache head dimension.
        self.key_projection = nn.Linear(
            embedding_dim,
            num_kv_heads * self.head_dim,
            bias=False,
        )

        self.value_projection = nn.Linear(
            embedding_dim,
            num_kv_heads * self.head_dim,
            bias=False,
        )

        self.output_projection = nn.Linear(
            num_query_heads * self.head_dim,
            embedding_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        past_key: torch.Tensor | None = None,
        past_value: torch.Tensor | None = None,
        use_cache: bool = False,
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
    ):
        """Apply grouped-query causal self-attention.

        Args:
            x: New input representations with shape (B, T_new, C).
            past_key: Compact cached keys with shape
                (B, H_KV, T_past, D), or None.
            past_value: Compact cached values with shape
                (B, H_KV, T_past, D), or None.
            use_cache: Whether the updated compact KV cache should be
                returned.

        Returns:
            If use_cache is false, attention output with shape
            (B, T_new, C). If use_cache is true, a tuple containing
            (output, key_cache, value_cache), where the caches keep shape
            (B, H_KV, T_total, D).

        Raises:
            ValueError: If input or cache shapes are inconsistent.
        """
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (B, T, C)."
            )

        batch_size, sequence_length, embedding_dim = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                "Input feature dimension must match embedding_dim."
            )

        # A valid cache contains both K and V, or neither.
        if (past_key is None) != (past_value is None):
            raise ValueError(
                "past_key and past_value must both be provided "
                "or both be None."
            )

        if past_key is None:
            past_length: int = 0
        else:
            if not use_cache:
                raise ValueError(
                    "use_cache must be True when a past cache "
                    "is provided."
                )

            if past_value is None:
                raise ValueError(
                    "past_value must be provided with past_key."
                )

            if past_key.ndim != 4:
                raise ValueError(
                    "past_key must have shape (B, H_KV, T_past, D)."
                )

            if past_key.shape != past_value.shape:
                raise ValueError(
                    "past_key and past_value must have identical shapes."
                )

            if past_key.shape[0] != batch_size:
                raise ValueError(
                    "Cache batch size must match input batch size."
                )

            if past_key.shape[1] != self.num_kv_heads:
                raise ValueError(
                    "Cache head count must match num_kv_heads."
                )

            if past_key.shape[-1] != self.head_dim:
                raise ValueError(
                    "Cache head dimension must match head_dim."
                )

            # This educational implementation decodes one new token at a time.
            if sequence_length != 1:
                raise ValueError(
                    "Cached decoding currently expects exactly one new token."
                )

            past_length = past_key.shape[2]

        # Compute Q/K/V only for tokens supplied in this forward call.
        q: torch.Tensor = self.query_projection(x)
        k: torch.Tensor = self.key_projection(x)
        v: torch.Tensor = self.value_projection(x)

        # Q uses H_Q heads.
        q = rearrange(
            q,
            "b t (h d) -> b h t d",
            h=self.num_query_heads,
        )

        # K/V remain compact with only H_KV heads.
        k = rearrange(
            k,
            "b t (h d) -> b h t d",
            h=self.num_kv_heads,
        )
        v = rearrange(
            v,
            "b t (h d) -> b h t d",
            h=self.num_kv_heads,
        )

        # Cached decoding starts after all positions already stored.
        cos_values, sin_values = build_rope_cos_sin(
            sequence_length=sequence_length,
            head_dim=self.head_dim,
            position_offset=past_length,
            base=self.rope_base,
            device=x.device,
            dtype=q.dtype,
        )

        q = apply_rope(
            q,
            cos_values,
            sin_values,
        )
        k = apply_rope(
            k,
            cos_values,
            sin_values,
        )

        # Store only the compact H_KV representation.
        if past_key is None:
            key_cache: torch.Tensor = k
            value_cache: torch.Tensor = v
        else:
            key_cache = torch.cat(
                [past_key, k],
                dim=2,
            )
            value_cache = torch.cat(
                [past_value, v],
                dim=2,
            )

        # Expand only for the attention computation. The persistent cache
        # remains (B, H_KV, T, D).
        k_expanded: torch.Tensor = expand_kv_heads(
            key_cache,
            num_query_heads=self.num_query_heads,
        )
        v_expanded: torch.Tensor = expand_kv_heads(
            value_cache,
            num_query_heads=self.num_query_heads,
        )

        # Full-sequence processing needs causal masking. During one-token
        # decoding, every cached key is already in the current causal past.
        is_causal: bool = past_key is None

        attended: torch.Tensor = F.scaled_dot_product_attention(
            q,
            k_expanded,
            v_expanded,
            is_causal=is_causal,
        )

        # Merge H_Q heads back into residual-stream width C.
        attended = rearrange(
            attended,
            "b h t d -> b t (h d)",
        )

        output: torch.Tensor = self.output_projection(
            attended
        )

        if use_cache:
            return (
                output,
                key_cache,
                value_cache,
            )

        return output


class CausalSelfAttention(GroupedQueryAttention):
    """Standard MHA compatibility wrapper.

    This class preserves the original API while delegating to the unified
    grouped-query implementation with H_KV == H_Q.

    Args:
        embedding_dim: Width of the residual stream.
        num_heads: Number of query and key/value heads.
        rope_base: Base controlling RoPE frequencies.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            num_query_heads=num_heads,
            num_kv_heads=num_heads,
            rope_base=rope_base,
        )
