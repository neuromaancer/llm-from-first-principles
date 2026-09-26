"""Transformer block used by the decoder-only language model."""

import torch
import torch.nn as nn

from .attention import CausalSelfAttention


class TransformerBlock(nn.Module):
    """Pre-norm decoder-only Transformer block with optional KV caching.

    Args:
        embedding_dim: Width of the residual stream.
        attention: Causal self-attention module.
        feed_forward: Token-wise feed-forward module.
    """

    def __init__(
        self,
        embedding_dim: int,
        attention: CausalSelfAttention,
        feed_forward: nn.Module,
    ) -> None:
        super().__init__()

        self.attention_norm = nn.RMSNorm(
            embedding_dim
        )
        self.attention = attention
        self.feed_forward_norm = nn.RMSNorm(
            embedding_dim
        )
        self.feed_forward = feed_forward

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
        """Apply one Transformer block.

        Args:
            x: Residual-stream tensor with shape (B, T_new, C).
            past_key: Cached keys for this layer, or None.
            past_value: Cached values for this layer, or None.
            use_cache: Whether the updated layer cache should be returned.

        Returns:
            If use_cache is false, the updated residual stream. If use_cache
            is true, a tuple (x, key_cache, value_cache).
        """
        normalized: torch.Tensor = self.attention_norm(
            x
        )

        attention_result = self.attention(
            normalized,
            past_key=past_key,
            past_value=past_value,
            use_cache=use_cache,
        )

        if use_cache:
            if not isinstance(
                attention_result,
                tuple,
            ):
                raise RuntimeError(
                    "Attention did not return a KV cache."
                )

            (
                attention_output,
                key_cache,
                value_cache,
            ) = attention_result
        else:
            if isinstance(
                attention_result,
                tuple,
            ):
                raise RuntimeError(
                    "Unexpected KV cache returned."
                )

            attention_output = attention_result

        # Attention writes one update into the residual stream.
        x = x + attention_output

        # The MLP is token-wise, so no historical MLP state is cached.
        x = x + self.feed_forward(
            self.feed_forward_norm(x)
        )

        if use_cache:
            return (
                x,
                key_cache,
                value_cache,
            )

        return x
