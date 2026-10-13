"""Masked PPO objectives and policy entropy from Lesson 21A."""

import torch

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute the mean over valid positions selected by a boolean mask."""
    if values.shape != mask.shape:
        raise ValueError("values and mask must have the same shape.")

    if mask.dtype != torch.bool:
        raise ValueError("mask must have boolean dtype.")

    # Boolean indexing keeps only valid positions.
    valid_values = values[mask]

    if valid_values.numel() == 0:
        raise ValueError("mask must select at least one value.")

    return valid_values.mean()


def ppo_policy_loss(
    current_log_probabilities: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    action_mask: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the PPO clipped policy loss and basic diagnostics."""
    if current_log_probabilities.shape != old_log_probabilities.shape:
        raise ValueError(
            "Current and old log-probabilities must have the same shape."
        )

    if current_log_probabilities.shape != advantages.shape:
        raise ValueError(
            "Log probabilities and advantages must have the same shape."
        )

    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip epsilon lie strictly between 0 and 1.")

    if action_mask.shape != advantages.shape:
        raise ValueError("action_mask must match advantages.")

    frozen_old_log_probabilities = old_log_probabilities.detach()

    frozen_advantages = advantages.detach()

    log_probability_ratios = (
        current_log_probabilities - frozen_old_log_probabilities
    )

    probability_ratios = torch.exp(log_probability_ratios)

    unclipped_objective = probability_ratios * frozen_advantages

    clipped_ratios = torch.clamp(
        probability_ratios, min=1.0 - clip_epsilon, max=1.0 + clip_epsilon
    )

    clipped_objective = clipped_ratios * frozen_advantages

    surrogate_objective = torch.minimum(unclipped_objective, clipped_objective)

    policy_loss = -masked_mean(
        surrogate_objective,
        action_mask,
    )

    with torch.no_grad():
        approximate_kl = masked_mean(
            probability_ratios - 1.0 - log_probability_ratios,
            action_mask,
        )

        outside_clip_region = (probability_ratios < 1.0 - clip_epsilon) | (
            probability_ratios > 1.0 + clip_epsilon
        )

        clip_fraction = masked_mean(
            outside_clip_region.to(dtype=probability_ratios.dtype),
            action_mask,
        )

    return policy_loss, approximate_kl, clip_fraction


def ppo_value_loss(
    current_values: torch.Tensor,
    value_targets: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the masked PPO value loss."""

    if current_values.shape != value_targets.shape:
        raise ValueError(
            "current_values and value_targets must have the same shape."
        )

    if current_values.shape != action_mask.shape:
        raise ValueError("action_mask must match the value tensors.")

    # Value targets are fixed during PPO optimization.
    frozen_targets = value_targets.detach()

    squared_errors = torch.square(current_values - frozen_targets)

    return 0.5 * masked_mean(squared_errors, action_mask)


def categorical_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute categorical entropy along the final vocabulary dimension."""

    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = torch.exp(log_probabilities)

    entropy = -torch.sum(probabilities * log_probabilities, dim=-1)

    return entropy


__all__ = ["categorical_entropy", "masked_mean", "ppo_policy_loss", "ppo_value_loss"]
