"""Critic-free group advantages and a per-response GRPO policy surrogate."""

import math

import torch


@torch.no_grad()
def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Center outcome rewards within each prompt, optionally by group std.

    Args:
        rewards: (P, G) finite floating-point rewards, with G >= 2.
        normalize: Divide centered rewards by their population standard
            deviation. False gives the centered rewards used by Dr. GRPO.
        epsilon: Positive denominator floor when normalize=True, bounded
            below by the dtype's smallest normal positive value.

    Returns:
        Detached (P, G) advantages. Constant-reward groups are exactly zero.
        Rewards belonging to different prompts are never mixed.
    """
    if rewards.ndim != 2 or rewards.shape[0] < 1 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape (P, G) with P >= 1 and G >= 2.")
    if not rewards.is_floating_point() or not torch.isfinite(rewards).all().item():
        raise ValueError("rewards must be finite floating-point values.")
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a Boolean.")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ValueError("epsilon must be finite and strictly positive.")

    # Equality is checked before reduction: repeated float32 0.1 can have
    # a rounded mean and a tiny nonzero computed standard deviation.
    constant_group = rewards.eq(rewards[:, :1]).all(dim=1, keepdim=True)
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    centered = centered.masked_fill(constant_group, 0.0)
    if not normalize:
        return centered

    scale = rewards.std(dim=1, unbiased=False, keepdim=True)
    # In float16, 1e-8 can round to zero. Constant groups must stay 0/1,
    # and other groups need a representable positive denominator.
    floor = max(epsilon, torch.finfo(rewards.dtype).tiny)
    denominator = scale.clamp_min(floor).masked_fill(constant_group, 1.0)
    return centered / denominator


def grpo_clipped_policy_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_advantages: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    clip_epsilon_high: float | None = None,
) -> torch.Tensor:
    """Compute GRPO's clipped policy term, averaging each response first.

    Args:
        current_log_probs: (P, G, T) differentiable selected-action logP.
        old_log_probs: (P, G, T) collection-time logP of those same actions.
        response_advantages: (P, G) outcome advantages shared over tokens.
        action_mask: (P, G, T) right-padded Boolean mask. Every response
            must have at least one real action.
        clip_epsilon: Lower clip distance, strictly between zero and one.
        clip_epsilon_high: Upper clip distance; defaults to clip_epsilon.
            A larger value permits the Clip-Higher experiment.

    Returns:
        Scalar negative surrogate: mean over real tokens per response,
        then mean over responses/prompts. Only current_log_probs receives
        gradients. This function does not add a reference KL penalty and
        is not a complete GRPO or DAPO training algorithm.
    """
    if (
        isinstance(clip_epsilon, bool)
        or not isinstance(clip_epsilon, (int, float))
        or not math.isfinite(clip_epsilon)
        or not 0 < clip_epsilon < 1
    ):
        raise ValueError("clip_epsilon must lie strictly between 0 and 1.")
    if clip_epsilon_high is None:
        clip_epsilon_high = clip_epsilon
    if (
        isinstance(clip_epsilon_high, bool)
        or not isinstance(clip_epsilon_high, (int, float))
        or not math.isfinite(clip_epsilon_high)
        or clip_epsilon_high <= 0
    ):
        raise ValueError("clip_epsilon_high must be finite and strictly positive.")
    if (
        current_log_probs.ndim != 3
        or any(size == 0 for size in current_log_probs.shape)
        or current_log_probs.shape[1] < 2
    ):
        raise ValueError("current_log_probs must have nonempty shape (P, G, T), G >= 2.")
    if old_log_probs.shape != current_log_probs.shape:
        raise ValueError("old_log_probs must match current_log_probs shape.")
    if response_advantages.shape != current_log_probs.shape[:2]:
        raise ValueError("response_advantages must have shape (P, G).")
    if action_mask.shape != current_log_probs.shape or action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be Boolean with shape (P, G, T).")

    tensors = (current_log_probs, old_log_probs, response_advantages)
    if not all(
        x.is_floating_point() and x.dtype == current_log_probs.dtype for x in tensors
    ):
        raise ValueError("Log-probs and advantages must have the same floating dtype.")
    if not all(x.device == current_log_probs.device for x in (*tensors, action_mask)):
        raise ValueError("All inputs must share a device.")
    if not all(torch.isfinite(x).all().item() for x in tensors):
        raise ValueError("All log-probs and advantages must be finite.")
    if (
        not action_mask[:, :, 0].all().item()
        or (action_mask[:, :, 1:] & ~action_mask[:, :, :-1]).any().item()
    ):
        raise ValueError("Each response needs nonempty, right-padded actions.")

    # Mask BEFORE exp: ignored extreme differences must not create inf/NaN
    # gradients that survive a later output mask.
    log_ratio = (current_log_probs - old_log_probs.detach()).masked_fill(
        ~action_mask, 0.0
    )
    ratio = log_ratio.exp()
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon_high)
    advantage = response_advantages.detach().unsqueeze(-1)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    response_scores = surrogate.masked_fill(~action_mask, 0.0).sum(dim=-1)
    response_scores = response_scores / action_mask.sum(dim=-1)
    return -response_scores.mean()


__all__ = ["group_relative_advantages", "grpo_clipped_policy_loss"]
