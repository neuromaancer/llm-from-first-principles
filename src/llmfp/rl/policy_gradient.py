"""Policy-gradient objectives shared by reinforcement-learning lessons."""

from typing import Literal

import torch


type Reduction = Literal["mean", "sum", "none"]


def reinforce_loss(
    log_probabilities: torch.Tensor,
    weights: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Build a REINFORCE-style surrogate policy loss.

    Args:
        log_probabilities: Log-probabilities of sampled actions. Each entry
            must correspond to the action whose credit is stored at the same
            position in `weights`.
        weights: Returns or advantage estimates used as scalar credit signals.
            They are detached because the policy-gradient estimator treats
            them as fixed weights for the policy update.
        reduction: How to aggregate per-decision losses: `mean`, `sum`, or
            `none`.

    Returns:
        Reduced loss tensor, or the unreduced per-decision loss when
        `reduction="none"`.

    Raises:
        ValueError: If shapes differ or the reduction is unsupported.
    """
    if log_probabilities.shape != weights.shape:
        raise ValueError(
            "log_probabilities and weights must have identical shapes."
        )

    per_decision_loss: torch.Tensor = -(
        weights.detach()
        * log_probabilities
    )

    if reduction == "none":
        return per_decision_loss

    if reduction == "sum":
        return per_decision_loss.sum()

    if reduction == "mean":
        return per_decision_loss.mean()

    raise ValueError(
        "reduction must be 'mean', 'sum', or 'none'."
    )


__all__ = [
    "Reduction",
    "reinforce_loss",
]
