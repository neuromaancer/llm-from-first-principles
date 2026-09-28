"""Kernelized causal linear attention with optional recurrent state."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

LinearAttentionState = tuple[
    torch.Tensor,
    torch.Tensor,
]


def positive_feature_map(
    x: torch.Tensor,
) -> torch.Tensor:
    """Apply the positive feature map ELU(x) + 1.

    The transform preserves the input shape and produces non-negative
    features suitable for the normalized kernel-attention formulation.

    Args:
        x: Input tensor of any shape.

    Returns:
        Feature tensor with the same shape as `x`.
    """
    return F.elu(x) + 1.0


class CausalLinearAttention(nn.Module):
    """Multi-head causal kernelized linear attention.

    This implementation uses the positive feature map

        phi(x) = ELU(x) + 1

    and represents causal history with two prefix states:

        S_t = sum_{j <= t} phi(k_j) v_j^T
        z_t = sum_{j <= t} phi(k_j)

    During full-sequence processing, all prefix states are constructed with
    cumulative sums. During recurrent inference, a previously returned final
    state can be supplied and extended by one token or by a short chunk.

    The implementation is intentionally educational rather than kernel
    optimized. In particular, full-sequence processing materializes prefix
    states with shape (B, H, T, D, D).

    Positional encoding is intentionally not built into this module. A
    compatible positional mechanism should be chosen explicitly before this
    module is integrated into the main GPT architecture.

    Args:
        embedding_dim: Width of the residual stream.
        num_heads: Number of linear-attention heads.
        epsilon: Small positive value used in the normalization denominator.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        if num_heads <= 0:
            raise ValueError(
                "num_heads must be positive."
            )

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        if epsilon <= 0:
            raise ValueError(
                "epsilon must be positive."
            )

        self.embedding_dim: int = embedding_dim
        self.num_heads: int = num_heads
        self.head_dim: int = (
            embedding_dim // num_heads
        )
        self.epsilon: float = epsilon

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

    def _validate_state(
        self,
        state: LinearAttentionState,
        batch_size: int,
    ) -> None:
        """Validate a recurrent linear-attention state.

        Args:
            state: Tuple `(kv_state, key_state)`.
            batch_size: Batch size of the current input.

        Raises:
            ValueError: If state shapes do not match this module.
        """
        kv_state, key_state = state

        expected_kv_shape = (
            batch_size,
            self.num_heads,
            self.head_dim,
            self.head_dim,
        )
        expected_key_shape = (
            batch_size,
            self.num_heads,
            self.head_dim,
        )

        if tuple(kv_state.shape) != expected_kv_shape:
            raise ValueError(
                "kv_state must have shape "
                f"{expected_kv_shape}, got {tuple(kv_state.shape)}."
            )

        if tuple(key_state.shape) != expected_key_shape:
            raise ValueError(
                "key_state must have shape "
                f"{expected_key_shape}, got {tuple(key_state.shape)}."
            )

    def forward(
        self,
        x: torch.Tensor,
        state: LinearAttentionState | None = None,
        use_state: bool = False,
    ) -> (
        torch.Tensor
        | tuple[
            torch.Tensor,
            LinearAttentionState,
        ]
    ):
        """Apply causal linear attention with optional recurrent state.

        Args:
            x: New residual-stream representations with shape
                `(B, T_new, C)`.
            state: Optional previous `(kv_state, key_state)`, where
                `kv_state` has shape `(B, H, D, D)` and `key_state`
                has shape `(B, H, D)`.
            use_state: Whether to return the final recurrent state after
                processing the supplied sequence or chunk.

        Returns:
            If `use_state` is false, attention output with shape
            `(B, T_new, C)`.

            If `use_state` is true, returns
            `(output, (kv_state, key_state))`, where the returned state
            summarizes all previous tokens plus the supplied chunk.

        Raises:
            ValueError: If input or state shapes are inconsistent.
        """
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (B, T, C)."
            )

        batch_size, sequence_length, embedding_dim = x.shape

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive."
            )

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                "Input feature dimension must match embedding_dim."
            )

        if state is not None:
            if not use_state:
                raise ValueError(
                    "use_state must be True when a previous state is provided."
                )

            self._validate_state(
                state,
                batch_size=batch_size,
            )

        q: torch.Tensor = self.query_projection(x)
        k: torch.Tensor = self.key_projection(x)
        v: torch.Tensor = self.value_projection(x)

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

        q_features: torch.Tensor = positive_feature_map(q)
        k_features: torch.Tensor = positive_feature_map(k)

        # Each token writes one outer-product update:
        #
        # (B, H, T, D, 1) * (B, H, T, 1, D)
        #     ->
        # (B, H, T, D, D)
        kv_updates: torch.Tensor = (
            k_features.unsqueeze(-1)
            * v.unsqueeze(-2)
        )

        # Build causal prefix states for every position in the supplied chunk.
        kv_prefix_states: torch.Tensor = torch.cumsum(
            kv_updates,
            dim=2,
        )
        key_prefix_states: torch.Tensor = torch.cumsum(
            k_features,
            dim=2,
        )

        if state is not None:
            previous_kv_state, previous_key_state = state

            # Every prefix in this chunk begins with the full previous state.
            kv_prefix_states = (
                kv_prefix_states
                + previous_kv_state.unsqueeze(2)
            )
            key_prefix_states = (
                key_prefix_states
                + previous_key_state.unsqueeze(2)
            )

        # Each query reads only from the prefix state available at its
        # position.
        #
        # (B, H, T, 1, D) @ (B, H, T, D, D)
        #     ->
        # (B, H, T, D)
        numerator: torch.Tensor = (
            q_features.unsqueeze(-2)
            @ kv_prefix_states
        ).squeeze(-2)

        # Normalization:
        #
        # (B, H, T, 1, D) @ (B, H, T, D, 1)
        #     ->
        # (B, H, T, 1)
        denominator: torch.Tensor = (
            q_features.unsqueeze(-2)
            @ key_prefix_states.unsqueeze(-1)
        ).squeeze(-1)

        attended: torch.Tensor = numerator / (
            denominator + self.epsilon
        )

        attended = rearrange(
            attended,
            "b h t d -> b t (h d)",
        )

        output: torch.Tensor = self.output_projection(
            attended
        )

        if use_state:
            final_state: LinearAttentionState = (
                kv_prefix_states[:, :, -1],
                key_prefix_states[:, :, -1],
            )

            return (
                output,
                final_state,
            )

        return output


__all__ = [
    "CausalLinearAttention",
    "LinearAttentionState",
    "positive_feature_map",
]
