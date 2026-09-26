"""Causal self-attention used by the decoder-only Transformer."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .rope import apply_rope, build_rope_cos_sin


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with RoPE and optional KV caching.

    During full-sequence training or prompt prefill, all supplied sequence
    positions are processed together. During cached decoding, only one new
    token is projected while previously computed keys and values are reused.

    Args:
        embedding_dim: Width of the residual stream.
        num_heads: Number of attention heads.
        rope_base: Base controlling the RoPE frequency range.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        rope_base: float = 10000.0,
    ) -> None:
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embedding_dim // num_heads
        self.rope_base: float = rope_base

        if self.head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE."
            )

        self.query_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )
        self.key_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )
        self.output_projection = nn.Linear(
            embedding_dim,
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
        """Apply causal self-attention with optional KV caching.

        Args:
            x: New input representations with shape (B, T_new, C).
            past_key: Cached keys with shape (B, H, T_past, D), or None.
            past_value: Cached values with shape (B, H, T_past, D), or None.
            use_cache: Whether updated key/value caches should be returned.

        Returns:
            If use_cache is false, attention output with shape
            (B, T_new, C). If use_cache is true, a tuple
            (output, key_cache, value_cache).

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
                "Input feature dimension does not match embedding_dim."
            )

        # A cache is valid only when both K and V are present.
        if (past_key is None) != (past_value is None):
            raise ValueError(
                "past_key and past_value must both be provided "
                "or both be None."
            )

        if past_key is not None:
            if not use_cache:
                raise ValueError(
                    "use_cache must be True when a past cache is provided."
                )

            if past_value is None:
                raise ValueError(
                    "past_value must be provided with past_key."
                )

            if past_key.ndim != 4:
                raise ValueError(
                    "past_key must have shape (B, H, T_past, D)."
                )

            if past_value.shape != past_key.shape:
                raise ValueError(
                    "past_key and past_value must have identical shapes."
                )

            if past_key.shape[0] != batch_size:
                raise ValueError(
                    "Cache batch size must match the input batch size."
                )

            if past_key.shape[1] != self.num_heads:
                raise ValueError(
                    "Cache head count must match num_heads."
                )

            if past_key.shape[-1] != self.head_dim:
                raise ValueError(
                    "Cache head dimension must match head_dim."
                )

            # This first cached implementation decodes one new token at a time.
            if sequence_length != 1:
                raise ValueError(
                    "Cached decoding currently expects exactly one new token."
                )

            past_length: int = past_key.shape[2]
        else:
            past_length = 0

        # Compute Q/K/V only for tokens supplied in this forward call.
        q: torch.Tensor = self.query_projection(x)
        k: torch.Tensor = self.key_projection(x)
        v: torch.Tensor = self.value_projection(x)

        # Split total model width C into H attention heads of width D.
        q = rearrange(
            q,
            "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        k = rearrange(
            k,
            "b t (h d) -> b h t d",
            h=self.num_heads,
        )
        v = rearrange(
            v,
            "b t (h d) -> b h t d",
            h=self.num_heads,
        )

        # Cached decoding starts after every position already in the cache.
        cos_values, sin_values = build_rope_cos_sin(
            sequence_length=sequence_length,
            head_dim=self.head_dim,
            position_offset=past_length,
            base=self.rope_base,
            device=x.device,
            dtype=q.dtype,
        )

        # Position information enters the attention geometry through Q and K.
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

        if past_key is None:
            # During training or prompt prefill, current K/V are the full cache.
            key_cache: torch.Tensor = k
            value_cache: torch.Tensor = v
        else:
            # Extend only the sequence axis with the newly computed K/V.
            key_cache = torch.cat(
                [past_key, k],
                dim=2,
            )
            value_cache = torch.cat(
                [past_value, v],
                dim=2,
            )

        # Full-sequence processing needs a causal mask. During one-token
        # decoding, every cached key is already in the current token's past.
        is_causal: bool = past_key is None

        attended: torch.Tensor = F.scaled_dot_product_attention(
            q,
            key_cache,
            value_cache,
            is_causal=is_causal,
        )

        # Merge H attention heads back into residual-stream width C.
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
