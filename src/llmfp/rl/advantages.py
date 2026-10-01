"""Advantage-estimation primitives for reinforcement learning."""

import torch


@torch.no_grad()
def td_error(
    rewards: torch.Tensor,
    current_values: torch.Tensor,
    next_values: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute one-step temporal-difference errors.

    The inputs may be scalars or tensors, but all four tensors must have the
    same shape. A bootstrap mask value of zero removes the next-state value
    from that transition.

    Args:
        rewards: Immediate observed rewards.
        current_values: Value estimates V(s_t).
        next_values: Value estimates V(s_{t+1}).
        bootstrap_mask: One where next-state bootstrapping is allowed and zero
            for true terminal transitions.
        gamma: Discount factor in the closed interval [0, 1].

    Returns:
        One-step TD errors with the same shape as the inputs.

    Raises:
        ValueError: If tensor shapes differ or gamma lies outside [0, 1].
    """
    if not (
        rewards.shape
        == current_values.shape
        == next_values.shape
        == bootstrap_mask.shape
    ):
        raise ValueError(
            "rewards, current_values, next_values, and bootstrap_mask "
            "must have identical shapes."
        )

    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            "gamma must be between 0 and 1."
        )

    mask: torch.Tensor = bootstrap_mask.to(
        dtype=rewards.dtype,
        device=rewards.device,
    )

    return (
        rewards
        + gamma
        * mask
        * next_values
        - current_values
    )


__all__ = [
    "td_error",
]
