"""Reference-policy KL diagnostics and frozen RLHF rollout reward shaping."""

import math

import torch
import torch.nn.functional as F


def categorical_kl_from_logits(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """Compute exact categorical KL at every supplied context position.

    Args:
        current_logits: (..., V) floating-point logits from the current policy.
        reference_logits: (..., V) logits from the reference policy, with
            identical shape, dtype, and device.

    Returns:
        (...) conditional forward KL values D_KL(current || reference).
        The operation preserves gradients. It does not average across states
        or estimate a whole-sequence, on-policy KL divergence. The caller
        chooses which contextual states to aggregate and usually freezes
        the reference network separately.

    Raises:
        ValueError: If tensor shape, dtype, device or entries are invalid.
    """
    if current_logits.ndim < 1 or any(size == 0 for size in current_logits.shape):
        raise ValueError("current_logits must have nonempty shape (..., V).")
    if reference_logits.shape != current_logits.shape:
        raise ValueError("reference_logits must match current_logits shape.")
    if (
        not current_logits.is_floating_point()
        or current_logits.dtype != reference_logits.dtype
    ):
        raise ValueError("Both logit tensors must have the same floating-point dtype.")
    if current_logits.device != reference_logits.device:
        raise ValueError("Both logit tensors must share a device.")
    if (
        not torch.isfinite(current_logits).all().item()
        or not torch.isfinite(reference_logits).all().item()
    ):
        raise ValueError("All logits must be finite.")

    # (..., V): conditional log-probabilities over the full vocabulary.
    current_log_probs = F.log_softmax(current_logits, dim=-1)
    reference_log_probs = F.log_softmax(reference_logits, dim=-1)
    # (...,): exact conditional forward KL, without state aggregation.
    return (
        current_log_probs.exp() * (current_log_probs - reference_log_probs)
    ).sum(dim=-1)


@torch.no_grad()
def kl_shaped_action_rewards(
    old_action_log_probs: torch.Tensor,
    reference_action_log_probs: torch.Tensor,
    action_mask: torch.Tensor,
    terminal_scores: torch.Tensor,
    terminated: torch.Tensor,
    *,
    beta: float,
) -> torch.Tensor:
    """Create frozen token KL rewards plus optional last-action terminal score.

    Args:
        old_action_log_probs: (B, T) actual on-policy sampling log-probs.
        reference_action_log_probs: (B, T) reference log-probs of the SAME
            sampled actions at the SAME prefixes.
        action_mask: (B, T) Boolean, right-padded mask of real actions;
            every row must contain at least one valid action.
        terminal_scores: (B,) reward-model scores for observed sequences.
        terminated: (B,) Boolean flags: true means the sequence terminated
            normally, false means it was truncated without terminal reward.
        beta: Finite, nonnegative coefficient on old-minus-reference log-ratios.

    Returns:
        (B, T) detached RLHF rewards. Each valid action receives
        -beta * (old_logp - reference_logp). For terminated rows, add the
        sequence-level score to the final valid action only. Padding stays
        zero and truncated rows receive no terminal score. This is a rollout
        preparation operation, not a differentiable current-policy KL loss.

    Raises:
        ValueError: If tensors, masks, or beta violate the contract.
    """
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and nonnegative.")
    if old_action_log_probs.ndim != 2 or any(s == 0 for s in old_action_log_probs.shape):
        raise ValueError("old_action_log_probs must have nonempty shape (B, T).")
    if reference_action_log_probs.shape != old_action_log_probs.shape:
        raise ValueError("reference_action_log_probs must match shape (B, T).")
    if (
        not old_action_log_probs.is_floating_point()
        or old_action_log_probs.dtype != reference_action_log_probs.dtype
    ):
        raise ValueError("Both log-probability tensors need the same floating dtype.")
    if action_mask.shape != old_action_log_probs.shape or action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be Boolean with shape (B, T).")
    batch_size = old_action_log_probs.shape[0]
    if terminal_scores.shape != (batch_size,) or terminal_scores.dtype != old_action_log_probs.dtype:
        raise ValueError("terminal_scores must be floating-point (B,) with matching dtype.")
    if terminated.shape != (batch_size,) or terminated.dtype != torch.bool:
        raise ValueError("terminated must be Boolean with shape (B,).")
    if any(
        x.device != old_action_log_probs.device
        for x in (reference_action_log_probs, action_mask, terminal_scores, terminated)
    ):
        raise ValueError("All tensors must share a device.")
    if not action_mask[:, 0].all().item() or (action_mask[:, 1:] & ~action_mask[:, :-1]).any().item():
        raise ValueError("Each row needs right-padded, nonempty valid actions.")
    if any(
        not torch.isfinite(x).all().item()
        for x in (old_action_log_probs, reference_action_log_probs, terminal_scores)
    ):
        raise ValueError("All log-probabilities and terminal scores must be finite.")

    # (B, T): signed sampled KL-shaped token rewards, zero on padding.
    action_rewards = (
        -beta * (old_action_log_probs - reference_action_log_probs)
    ).masked_fill(~action_mask, 0.0)

    # (B,): final valid action index per episode, excluding padding.
    final_action_indices = action_mask.sum(dim=1) - 1
    # (B_terminated,): rows that ended normally; truncated rows are skipped.
    terminal_rows = torch.arange(batch_size, device=action_mask.device)[terminated]
    # Add the learned terminal reward exactly once for normally ended episodes.
    action_rewards[
        terminal_rows, final_action_indices[terminated]
    ] += terminal_scores[terminated]
    return action_rewards


__all__ = ["categorical_kl_from_logits", "kl_shaped_action_rewards"]
