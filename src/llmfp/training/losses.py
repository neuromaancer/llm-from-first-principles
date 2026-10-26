"""Supervised-token losses derived and verified in Lesson 22."""

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


__all__ = ["per_example_sft_nll"]
