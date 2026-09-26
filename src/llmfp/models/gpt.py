"""Decoder-only GPT model used by later lessons."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from llmfp.generation.cache import KVCache
from llmfp.nn.attention import GroupedQueryAttention
from llmfp.nn.mlp import SwiGLU
from llmfp.nn.transformer import TransformerBlock


@dataclass
class GPTConfig:
    """Configuration for the decoder-only Transformer.

    Attributes:
        vocab_size: Number of tokens in the vocabulary.
        embedding_dim: Width of the residual stream.
        num_query_heads: Number of query heads per attention layer.
        num_kv_heads: Number of key/value heads per attention layer.
            Setting this equal to num_query_heads gives MHA; setting it
            to one gives MQA; intermediate values give GQA.
        num_layers: Number of stacked Transformer blocks.
        hidden_dim: Intermediate width of each SwiGLU network.
        rope_base: Base controlling the RoPE frequency range.
    """

    vocab_size: int
    embedding_dim: int
    num_query_heads: int
    num_kv_heads: int
    num_layers: int
    hidden_dim: int
    rope_base: float = 10000.0

    def __post_init__(self) -> None:
        """Validate architectural relationships."""
        if self.vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive."
            )

        if self.embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        if self.num_query_heads <= 0:
            raise ValueError(
                "num_query_heads must be positive."
            )

        if self.num_kv_heads <= 0:
            raise ValueError(
                "num_kv_heads must be positive."
            )

        if self.num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if self.hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if self.embedding_dim % self.num_query_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_query_heads."
            )

        if self.num_query_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_query_heads must be divisible by num_kv_heads."
            )

        head_dim: int = (
            self.embedding_dim // self.num_query_heads
        )

        if head_dim % 2 != 0:
            raise ValueError(
                "head_dim must be even for RoPE."
            )


class GPT(nn.Module):
    """Decoder-only Transformer language model with optional KV caching.

    Args:
        config: Model architecture configuration.
    """

    def __init__(
        self,
        config: GPTConfig,
    ) -> None:
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    attention=GroupedQueryAttention(
                        embedding_dim=config.embedding_dim,
                        num_query_heads=config.num_query_heads,
                        num_kv_heads=config.num_kv_heads,
                        rope_base=config.rope_base,
                    ),
                    feed_forward=SwiGLU(
                        embedding_dim=config.embedding_dim,
                        hidden_dim=config.hidden_dim,
                    ),
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.RMSNorm(
            config.embedding_dim
        )

        self.lm_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        token_ids: torch.Tensor,
        past_key_values: KVCache | None = None,
        use_cache: bool = False,
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor,
            KVCache,
        ]
    ):
        """Convert token IDs into logits with optional KV caching.

        Args:
            token_ids: New token IDs with shape (B, T_new).
            past_key_values: One compact (K, V) pair per Transformer
                layer, or None.
            use_cache: Whether updated per-layer caches should be returned.

        Returns:
            If use_cache is false, vocabulary logits with shape
            (B, T_new, V). If use_cache is true, a tuple
            (logits, new_cache).

        Raises:
            ValueError: If token IDs or the cache structure are invalid.
        """
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape (B, T)."
            )

        if past_key_values is not None:
            if not use_cache:
                raise ValueError(
                    "use_cache must be True when past_key_values "
                    "are provided."
                )

            if len(past_key_values) != len(
                self.blocks
            ):
                raise ValueError(
                    "past_key_values must contain exactly one cache "
                    "pair per Transformer block."
                )

        # Convert discrete token IDs into residual-stream vectors.
        x: torch.Tensor = self.token_embedding(
            token_ids
        )

        new_cache: KVCache = []

        for layer_index, block in enumerate(
            self.blocks
        ):
            if past_key_values is None:
                past_key = None
                past_value = None
            else:
                (
                    past_key,
                    past_value,
                ) = past_key_values[layer_index]

            block_result = block(
                x,
                past_key=past_key,
                past_value=past_value,
                use_cache=use_cache,
            )

            if use_cache:
                if not isinstance(
                    block_result,
                    tuple,
                ):
                    raise RuntimeError(
                        "Transformer block did not return a KV cache."
                    )

                (
                    x,
                    key_cache,
                    value_cache,
                ) = block_result

                # Cache order matches Transformer layer order.
                new_cache.append(
                    (
                        key_cache,
                        value_cache,
                    )
                )
            else:
                if isinstance(
                    block_result,
                    tuple,
                ):
                    raise RuntimeError(
                        "Unexpected KV cache returned."
                    )

                x = block_result

        x = self.final_norm(x)

        # Produce one vocabulary-logit vector per supplied token position.
        logits: torch.Tensor = self.lm_head(
            x
        )

        if use_cache:
            return (
                logits,
                new_cache,
            )

        return logits
