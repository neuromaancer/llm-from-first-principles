"""Sparse mixture-of-experts feed-forward modules."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import SwiGLU


def expert_utilization(
    selected_experts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Measure the fraction of routes assigned to each expert.

    Args:
        selected_experts: Selected expert IDs with shape (N, K).
        num_experts: Total number of experts.

    Returns:
        Route fractions with shape (E) that sum to one.
    """
    if selected_experts.ndim != 2:
        raise ValueError(
            "selected_experts must have shape (N, K)."
        )

    if num_experts <= 0:
        raise ValueError(
            "num_experts must be positive."
        )

    total_routes: int = selected_experts.numel()

    if total_routes == 0:
        raise ValueError(
            "selected_experts must contain at least one route."
        )

    fractions: list[torch.Tensor] = []

    for expert_index in range(num_experts):
        route_fraction: torch.Tensor = (
            (selected_experts == expert_index)
            .to(dtype=torch.float32)
            .mean()
        )

        fractions.append(
            route_fraction
        )

    return torch.stack(
        fractions,
        dim=0,
    )


def load_balancing_loss(
    router_probabilities: torch.Tensor,
    selected_experts: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """Encourage router probability mass to follow balanced utilization.

    This educational objective combines hard route fractions with the
    differentiable mean router probability assigned to each expert.

    Args:
        router_probabilities: Full router probabilities with shape (N, E).
        selected_experts: Selected expert IDs with shape (N, K).
        num_experts: Total number of experts.

    Returns:
        Scalar auxiliary load-balancing loss.
    """
    if router_probabilities.ndim != 2:
        raise ValueError(
            "router_probabilities must have shape (N, E)."
        )

    if selected_experts.ndim != 2:
        raise ValueError(
            "selected_experts must have shape (N, K)."
        )

    if router_probabilities.shape[0] != selected_experts.shape[0]:
        raise ValueError(
            "router probabilities and selections must contain the same tokens."
        )

    if router_probabilities.shape[1] != num_experts:
        raise ValueError(
            "router probability width must match num_experts."
        )

    utilization: torch.Tensor = expert_utilization(
        selected_experts,
        num_experts=num_experts,
    ).to(
        dtype=router_probabilities.dtype,
        device=router_probabilities.device,
    )

    mean_router_probability: torch.Tensor = (
        router_probabilities.mean(
            dim=0
        )
    )

    return (
        num_experts
        * torch.sum(
            utilization
            * mean_router_probability
        )
    )


def expert_capacity(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    capacity_factor: float,
) -> int:
    """Compute a simple per-expert route capacity.

    Args:
        num_tokens: Number of token representations.
        top_k: Number of selected experts per token.
        num_experts: Total number of experts.
        capacity_factor: Extra capacity relative to perfectly balanced load.

    Returns:
        Maximum number of routes accepted by each expert.
    """
    if num_tokens <= 0:
        raise ValueError(
            "num_tokens must be positive."
        )

    if num_experts <= 0:
        raise ValueError(
            "num_experts must be positive."
        )

    if not 1 <= top_k <= num_experts:
        raise ValueError(
            "top_k must be between 1 and num_experts."
        )

    if capacity_factor <= 0.0:
        raise ValueError(
            "capacity_factor must be positive."
        )

    average_routes_per_expert: float = (
        num_tokens
        * top_k
        / num_experts
    )

    return math.ceil(
        capacity_factor
        * average_routes_per_expert
    )


class SparseMoE(nn.Module):
    """Educational sparse top-k mixture of SwiGLU experts.

    The module keeps the same external shape as a dense Transformer FFN:
    (B, T, C) -> (B, T, C). Routing and dispatch are intentionally explicit
    rather than optimized so the token-to-expert data movement remains easy
    to inspect.

    Args:
        embedding_dim: Width of the residual stream.
        hidden_dim: Hidden width of each expert.
        num_experts: Number of available experts.
        top_k: Number of experts selected per token.
        capacity_factor: Optional expert-capacity multiplier. When omitted,
            every requested route is accepted.
        renormalize_selected_weights: Whether selected routing probabilities
            are renormalized to sum to one. With top-1 routing, renormalizing
            makes the single selected weight exactly one, so the task loss
            does not train the router through that output gate.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        capacity_factor: float | None = None,
        renormalize_selected_weights: bool = True,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive."
            )

        if hidden_dim <= 0:
            raise ValueError(
                "hidden_dim must be positive."
            )

        if num_experts <= 0:
            raise ValueError(
                "num_experts must be positive."
            )

        if not 1 <= top_k <= num_experts:
            raise ValueError(
                "top_k must be between 1 and num_experts."
            )

        if (
            capacity_factor is not None
            and capacity_factor <= 0.0
        ):
            raise ValueError(
                "capacity_factor must be positive when provided."
            )

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.renormalize_selected_weights = renormalize_selected_weights

        self.router = nn.Linear(
            embedding_dim,
            num_experts,
            bias=False,
        )

        self.experts = nn.ModuleList(
            [
                SwiGLU(
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Apply sparse MoE and return only token updates."""
        output, _, _ = self.forward_with_auxiliary(
            x
        )

        return output

    def forward_with_auxiliary(
        self,
        x: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Apply sparse MoE and expose routing diagnostics.

        Args:
            x: Token representations with shape (B, T, C).

        Returns:
            Tuple containing the MoE output, scalar balancing loss, and a
            dictionary of routing diagnostics.
        """
        if x.ndim != 3:
            raise ValueError(
                "x must have shape (B, T, C)."
            )

        (
            batch_size,
            sequence_length,
            embedding_dim,
        ) = x.shape

        if embedding_dim != self.embedding_dim:
            raise ValueError(
                "Input width must match embedding_dim."
            )

        num_tokens: int = (
            batch_size
            * sequence_length
        )

        x_flat: torch.Tensor = x.reshape(
            num_tokens,
            embedding_dim,
        )

        router_logits: torch.Tensor = self.router(
            x_flat
        )

        router_probabilities: torch.Tensor = F.softmax(
            router_logits,
            dim=-1,
        )

        (
            topk_probabilities,
            topk_indices,
        ) = torch.topk(
            router_probabilities,
            k=self.top_k,
            dim=-1,
        )

        if self.renormalize_selected_weights:
            topk_weights: torch.Tensor = (
                topk_probabilities
                / topk_probabilities.sum(
                    dim=-1,
                    keepdim=True,
                )
            )
        else:
            # Keeping the raw selected probabilities preserves the router's
            # absolute confidence. This is especially important for top-1
            # routing, where renormalization would make every gate equal 1.
            topk_weights = topk_probabilities

        balance_loss: torch.Tensor = load_balancing_loss(
            router_probabilities,
            topk_indices,
            num_experts=self.num_experts,
        )

        utilization: torch.Tensor = expert_utilization(
            topk_indices,
            num_experts=self.num_experts,
        )

        accepted_route_mask: torch.Tensor = torch.ones_like(
            topk_indices,
            dtype=torch.bool,
        )

        if self.capacity_factor is not None:
            capacity: int = expert_capacity(
                num_tokens=num_tokens,
                top_k=self.top_k,
                num_experts=self.num_experts,
                capacity_factor=self.capacity_factor,
            )

            accepted_route_mask = torch.zeros_like(
                topk_indices,
                dtype=torch.bool,
            )

            for expert_index in range(
                self.num_experts
            ):
                routes: list[
                    tuple[int, int, float]
                ] = []

                for token_index in range(
                    num_tokens
                ):
                    for route_position in range(
                        self.top_k
                    ):
                        selected_expert: int = int(
                            topk_indices[
                                token_index,
                                route_position,
                            ].item()
                        )

                        if selected_expert != expert_index:
                            continue

                        route_weight: float = float(
                            topk_weights[
                                token_index,
                                route_position,
                            ]
                            .detach()
                            .item()
                        )

                        routes.append(
                            (
                                token_index,
                                route_position,
                                route_weight,
                            )
                        )

                routes.sort(
                    key=lambda route: route[2],
                    reverse=True,
                )

                for (
                    token_index,
                    route_position,
                    _,
                ) in routes[:capacity]:
                    accepted_route_mask[
                        token_index,
                        route_position,
                    ] = True

        expert_routes: list[
            list[tuple[int, int]]
        ] = [
            []
            for _ in range(
                self.num_experts
            )
        ]

        for token_index in range(
            num_tokens
        ):
            for route_position in range(
                self.top_k
            ):
                if not bool(
                    accepted_route_mask[
                        token_index,
                        route_position,
                    ].item()
                ):
                    continue

                expert_index: int = int(
                    topk_indices[
                        token_index,
                        route_position,
                    ].item()
                )

                expert_routes[
                    expert_index
                ].append(
                    (
                        token_index,
                        route_position,
                    )
                )

        token_contributions: list[
            list[torch.Tensor]
        ] = [
            []
            for _ in range(
                num_tokens
            )
        ]

        for expert_index, routes in enumerate(
            expert_routes
        ):
            if not routes:
                continue

            token_indices: list[int] = [
                token_index
                for token_index, _
                in routes
            ]

            route_positions: list[int] = [
                route_position
                for _, route_position
                in routes
            ]

            token_index_tensor = torch.tensor(
                token_indices,
                dtype=torch.long,
                device=x.device,
            )

            route_position_tensor = torch.tensor(
                route_positions,
                dtype=torch.long,
                device=x.device,
            )

            expert_inputs: torch.Tensor = x_flat[
                token_index_tensor
            ]

            expert_outputs: torch.Tensor = self.experts[
                expert_index
            ](
                expert_inputs
            )

            routing_weights: torch.Tensor = (
                topk_weights[
                    token_index_tensor,
                    route_position_tensor,
                ]
                .unsqueeze(-1)
            )

            weighted_outputs: torch.Tensor = (
                routing_weights
                * expert_outputs
            )

            for (
                local_index,
                token_index,
            ) in enumerate(
                token_indices
            ):
                token_contributions[
                    token_index
                ].append(
                    weighted_outputs[
                        local_index
                    ]
                )

        output_tokens: list[
            torch.Tensor
        ] = []

        for token_index, contributions in enumerate(
            token_contributions
        ):
            if contributions:
                token_output: torch.Tensor = torch.stack(
                    contributions,
                    dim=0,
                ).sum(
                    dim=0
                )
            else:
                token_output = torch.zeros_like(
                    x_flat[
                        token_index
                    ]
                )

            output_tokens.append(
                token_output
            )

        output_flat: torch.Tensor = torch.stack(
            output_tokens,
            dim=0,
        )

        output: torch.Tensor = output_flat.reshape(
            batch_size,
            sequence_length,
            embedding_dim,
        )

        routing_info: dict[str, torch.Tensor] = {
            "router_probabilities": router_probabilities,
            "selected_experts": topk_indices,
            "selected_weights": topk_weights,
            "utilization": utilization,
            "accepted_route_mask": accepted_route_mask,
        }

        return (
            output,
            balance_loss,
            routing_info,
        )


__all__ = [
    "SparseMoE",
    "expert_capacity",
    "expert_utilization",
    "load_balancing_loss",
]
