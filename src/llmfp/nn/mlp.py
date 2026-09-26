"""Feed-forward modules used by Transformer blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network for a Transformer block.

    The module uses two parallel input projections. One branch is passed
    through SiLU and acts as a learned gate for the other branch.

    Args:
        embedding_dim: Width of the residual stream.
        hidden_dim: Width of the intermediate feed-forward representation.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()

        self.gate_projection = nn.Linear(
            embedding_dim,
            hidden_dim,
            bias=False,
        )
        self.up_projection = nn.Linear(
            embedding_dim,
            hidden_dim,
            bias=False,
        )
        self.down_projection = nn.Linear(
            hidden_dim,
            embedding_dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the SwiGLU transformation.

        Args:
            x: Input tensor with shape (B, T, C).

        Returns:
            Output tensor with the same shape (B, T, C).
        """
        # The gate branch learns how strongly each hidden feature should
        # contribute to the output.
        gate: torch.Tensor = F.silu(
            self.gate_projection(x)
        )

        # The up branch contains the candidate hidden features.
        content: torch.Tensor = self.up_projection(x)

        # Element-wise multiplication performs learned feature gating.
        hidden: torch.Tensor = gate * content

        # Project back to the residual-stream width.
        return self.down_projection(hidden)
