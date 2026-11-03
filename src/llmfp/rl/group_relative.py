"""Critic-free group advantages and the original GRPO clipped policy surrogate."""

import math

import torch


@torch.no_grad()
def group_relative_advantages(
    rewards: torch.Tensor,
    *,
    normalize: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Center completion rewards within each prompt, optionally by group std.

    Args:
        rewards: (P, G) finite floating-point outcome rewards. P prompts;
            G >= 2 independently sampled completions per prompt.
        normalize: True for original GRPO's population-std normalization;
            False for the unstandardized group-centered Dr. GRPO variant.
        epsilon: Positive denominator floor when normalize=True.

    Returns:
        (P, G) frozen advantages. Equal-reward groups yield exactly zero.
        A shared prompt's rewards are never mixed with another prompt's.
    """
    if rewards.ndim != 2 or rewards.shape[0] < 1 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape (P, G) with P >= 1 and G >= 2.")
    if not rewards.is_floating_point() or not torch.isfinite(rewards).all().item():
        raise ValueError("rewards must be finite floating-point values.")
    if not isinstance(normalize, bool):
        raise ValueError("normalize must be a Boolean.")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and strictly positive.")

    # (P, G): per-prompt, mean-centered outcome rewards.
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    if not normalize:
        return centered
    # (P, 1): population standard deviation of each prompt's group.
    scale = rewards.std(dim=1, unbiased=False, keepdim=True)
    return centered / scale.clamp_min(epsilon)


def grpo_clipped_policy_loss(
    current_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    response_advantages: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.2,
    clip_epsilon_high: float | None = None,
) -> torch.Tensor:
    """Return the per-response-normalized GRPO clipped policy loss.

    Args:
        current_log_probs: (P, G, T) trainable selected-action log-probs.
        old_log_probs: (P, G, T) fixed behavior-policy log-probs, same actions.
        response_advantages: (P, G) group-relative outcome advantages.
        action_mask: (P, G, T) right-padded Boolean mask for generated tokens.
            Each response must have at least one valid action.
        clip_epsilon: Positive lower clip distance, strictly less than one.
        clip_epsilon_high: Positive upper clip distance; defaults to the same
            value as clip_epsilon. A larger value models Clip-Higher.

    Returns:
        Scalar negative clipped surrogate: first average valid tokens
        within each response, then average responses and prompts. Gradients
        flow only through current_log_probs. Excludes reference KL, reward
        scoring, and group-advantage construction.
    """
    if isinstance(clip_epsilon, bool) or not isinstance(clip_epsilon, (int, float)) or not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
        raise ValueError("clip_epsilon must lie strictly between 0 and 1.")
    if clip_epsilon_high is None:
        clip_epsilon_high = clip_epsilon
    if isinstance(clip_epsilon_high, bool) or not isinstance(clip_epsilon_high, (int, float)) or not math.isfinite(clip_epsilon_high) or clip_epsilon_high <= 0:
        raise ValueError("clip_epsilon_high must be finite and strictly positive.")
    if current_log_probs.ndim != 3 or any(size == 0 for size in current_log_probs.shape) or current_log_probs.shape[1] < 2:
        raise ValueError("current_log_probs must have nonempty shape (P, G, T), G >= 2.")
    if old_log_probs.shape != current_log_probs.shape:
        raise ValueError("old_log_probs must match current_log_probs shape.")
    if response_advantages.shape != current_log_probs.shape[:2]:
        raise ValueError("response_advantages must have shape (P, G).")
    if action_mask.shape != current_log_probs.shape or action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be Boolean with shape (P, G, T).")
    tensors = (current_log_probs, old_log_probs, response_advantages)
    if not all(x.is_floating_point() and x.dtype == current_log_probs.dtype for x in tensors):
        raise ValueError("Log-probs and advantages must have the same floating dtype.")
    if not all(x.device == current_log_probs.device for x in (*tensors, action_mask)):
        raise ValueError("All inputs must share a device.")
    if not all(torch.isfinite(x).all().item() for x in tensors):
        raise ValueError("All log-probs and advantages must be finite.")
    if not action_mask[:, :, 0].all().item() or (action_mask[:, :, 1:] & ~action_mask[:, :, :-1]).any().item():
        raise ValueError("Each response needs nonempty, right-padded actions.")

    # (P, G, T): importance ratios against collection-time policy.
    ratio = (current_log_probs - old_log_probs.detach()).exp()
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon_high)
    # (P, G, 1): outcome advantage shared across this response's tokens.
    advantage = response_advantages.detach().unsqueeze(-1)
    clipped_surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    # (P, G): each response contributes equally, regardless of token length.
    per_response = clipped_surrogate.masked_fill(~action_mask, 0.0).sum(dim=-1) / action_mask.sum(dim=-1)
    return -per_response.mean()


__all__ = ["group_relative_advantages", "grpo_clipped_policy_loss"]
