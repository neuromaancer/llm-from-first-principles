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



@torch.no_grad()
def generalized_advantage_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    continuation_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """Compute GAE with separate bootstrap and continuation semantics."""

    if rewards.ndim != 1:
        raise ValueError("rewards must have shape (T,).")

    if values.ndim != 1:
        raise ValueError("values must have shape (T + 1,).")

    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError("values must contain one more element than rewards.")

    if bootstrap_mask.shape != rewards.shape:
        raise ValueError("bootstrap_mask must match rewards.")

    if continuation_mask.shape != rewards.shape:
        raise ValueError("continuation_mask must match rewards.")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must lie between 0 and 1.")

    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError("gae_lambda must lie between 0 and 1.")

    if rewards.numel() == 0:
        raise ValueError("GAE requires at least one transition.")

    if not (rewards.device == values.device == bootstrap_mask.device == continuation_mask.device):
        raise ValueError("GAE inputs must be on the same device.")

    if not torch.is_floating_point(rewards) or not torch.is_floating_point(values):
        raise ValueError("Rewards and values must be floating-point tensors.")

    for name, mask in (("bootstrap_mask", bootstrap_mask), ("continuation_mask", continuation_mask)):
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError(f"{name} must contain only 0 or 1.")

    if bool(((continuation_mask == 1) & (bootstrap_mask == 0)).any()):
        raise ValueError("Continuation is invalid when bootstrapping is disabled.")

    current_values = values[:-1]
    next_values = values[1:]

    bootstrap_mask = bootstrap_mask.to(dtype=values.dtype, device=values.device)

    continuation_mask = continuation_mask.to(
        dtype=values.dtype, device=values.device
    )

    td_errors = td_error(rewards, current_values, next_values, bootstrap_mask, gamma)

    advantages = torch.zeros_like(rewards)

    running_advantage = torch.zeros(
        (), dtype=rewards.dtype, device=rewards.device
    )

    for index in range(rewards.shape[0] - 1, -1, -1):
        running_advantage = (
            td_errors[index]
            + gamma * gae_lambda * continuation_mask[index] * running_advantage
        )

        advantages[index] = running_advantage

    value_targets = advantages + current_values

    return (advantages, value_targets)


__all__ = ["generalized_advantage_estimation", "td_error"]
