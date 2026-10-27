"""Supervised and preference losses for LLM post-training."""

from typing import Literal

import torch
import torch.nn.functional as F


def per_example_sft_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Return one mean supervised-token NLL per sequence without detaching.

    Args:
        logits: (B, T, V) floating-point next-token predictions.
        labels: (B, T) int64 targets ALREADY aligned with logits. Unsupervised
            positions contain ignore_index. No additional shift is performed.
        ignore_index: Negative label sentinel; never an input token ID.

    Returns:
        (B,) differentiable mean NLLs. With n_i supervised targets per row,
        the token-average batch objective is sum(n_i * loss_i) / sum(n_i).
        Taking result.mean() instead weights examples equally. Neither class
        weighting nor label smoothing is applied by this educational helper.

    Raises:
        ValueError: If shapes, dtypes, devices, labels, or target counts are invalid.
    """
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool) or ignore_index >= 0:
        raise ValueError("ignore_index must be a negative integer.")
    if logits.ndim != 3 or any(size == 0 for size in logits.shape):
        raise ValueError("logits must have nonempty shape (B, T, V).")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating-point.")
    if labels.shape != logits.shape[:2] or labels.dtype != torch.long:
        raise ValueError("labels must be int64 with shape (B, T).")
    if labels.device != logits.device:
        raise ValueError("logits and labels must share a device.")

    # (B, T): select positions whose targets contribute to supervision.
    mask = labels.ne(ignore_index)
    # (B,): denominator for each sequence's own mean NLL.
    counts = mask.sum(dim=1)
    if (counts == 0).any().item():
        raise ValueError("Every example needs at least one supervised target.")
    vocab_size = logits.shape[-1]
    if (mask & ((labels < 0) | (labels >= vocab_size))).any().item():
        raise ValueError("Supervised labels must lie in [0, V).")

    # (B, T): unreduced NLL; ignored positions have zero direct contribution.
    token_nll = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(labels.shape)
    return token_nll.sum(dim=1) / counts



def completion_sequence_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Score selected completion targets under teacher forcing.

    Args:
        logits: (B, T, V) causal next-token logits.
        labels: (B, T) int64 next-token targets ALREADY aligned with logits;
            prompt and right-padding targets must be set to ignore_index.
        ignore_index: Negative sentinel for unscored target positions.

    Returns:
        (B,) sum of token log-probabilities over scored completion targets.
        Values include a completion terminator only if the caller selected it.
        No additional shift or length normalization is performed. Gradients
        are preserved for policy-optimization objectives.

    Raises:
        ValueError: If tensor shape, dtype, device, or supervision is invalid.
    """
    if (
        not isinstance(ignore_index, int)
        or isinstance(ignore_index, bool)
        or ignore_index >= 0
    ):
        raise ValueError("ignore_index must be a negative integer.")
    if logits.ndim != 3 or any(size == 0 for size in logits.shape):
        raise ValueError("logits must have nonempty shape (B, T, V).")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating-point.")
    if labels.shape != logits.shape[:2] or labels.dtype != torch.long:
        raise ValueError("labels must be int64 with shape (B, T).")
    if labels.device != logits.device:
        raise ValueError("logits and labels must share a device.")

    # (B, T): supervision mask already aligned to the logits.
    mask = labels.ne(ignore_index)
    if not mask.any(dim=1).all().item():
        raise ValueError("Every example needs at least one completion target.")
    vocab_size = logits.shape[-1]
    if (mask & ((labels < 0) | (labels >= vocab_size))).any().item():
        raise ValueError("Supervised labels must lie in [0, V).")

    # (B, T, V): normalized log-probabilities over the vocabulary axis.
    all_log_probs = F.log_softmax(logits, dim=-1)
    # (B, T): vocabulary-safe indices; ignored labels never reach gather.
    safe_labels = labels.masked_fill(~mask, 0)
    # (B, T): log-probability of each actual next-token target.
    selected = all_log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    # (B,): sum over the caller-selected completion targets, not a mean.
    return selected.masked_fill(~mask, 0.0).sum(dim=1)


def bradley_terry_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    *,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> torch.Tensor:
    """Return stable pairwise negative log-likelihood from reward differences.

    Args:
        chosen_rewards: (B,) scalar rewards for preferred completions.
        rejected_rewards: (B,) scalar rewards for less-preferred completions.
            Both vectors must share the same dtype, shape, and device.
        reduction: 'none' returns (B,) individual pair losses; 'mean' returns
            the equal-pair-weighted scalar objective; 'sum' returns their sum.

    Returns:
        (B,) or () differentiable values softplus(rejected - chosen).
        These are Bradley-Terry rewards, not policy log-probabilities.

    Raises:
        ValueError: If shapes, dtypes, devices, finiteness or reduction is invalid.
    """
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'.")
    if chosen_rewards.ndim != 1 or chosen_rewards.numel() == 0:
        raise ValueError("chosen_rewards must have nonempty shape (B,).")
    if rejected_rewards.shape != chosen_rewards.shape:
        raise ValueError("rejected_rewards must match chosen_rewards shape.")
    if (
        not chosen_rewards.is_floating_point()
        or chosen_rewards.dtype != rejected_rewards.dtype
    ):
        raise ValueError("Both reward vectors must have the same floating-point dtype.")
    if chosen_rewards.device != rejected_rewards.device:
        raise ValueError("Both reward vectors must share a device.")
    if (
        not torch.isfinite(chosen_rewards).all().item()
        or not torch.isfinite(rejected_rewards).all().item()
    ):
        raise ValueError("Reward vectors must contain only finite values.")

    # (B,): loss for each observed chosen-over-rejected preference.
    losses = F.softplus(rejected_rewards - chosen_rewards)
    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


__all__ = [
    "bradley_terry_loss",
    "completion_sequence_log_probs",
    "per_example_sft_nll",
]
