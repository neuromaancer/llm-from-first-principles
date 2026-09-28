"""Single-step optimization helpers for language-model training."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TrainingStepResult:
    """Metrics produced by one language-model optimization step."""

    loss: float
    gradient_norm: float | None


def language_model_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_grad_norm: float | None = None,
) -> TrainingStepResult:
    """Run one next-token language-model optimization step.

    Args:
        model: Language model returning logits with shape (B, T, V).
        optimizer: PyTorch optimizer responsible for model parameters.
        inputs: Token IDs with shape (B, T).
        targets: Next-token targets with shape (B, T).
        max_grad_norm: Optional global gradient-norm clipping threshold.

    Returns:
        Loss and the pre-clipping global gradient norm when clipping is used.

    Raises:
        ValueError: If input/target shapes differ or max_grad_norm is invalid.
        RuntimeError: If the model unexpectedly returns a cache tuple.
    """
    if inputs.shape != targets.shape:
        raise ValueError(
            "inputs and targets must have identical shapes."
        )

    if max_grad_norm is not None and max_grad_norm <= 0.0:
        raise ValueError(
            "max_grad_norm must be positive when provided."
        )

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    logits = model(inputs)

    if not isinstance(
        logits,
        torch.Tensor,
    ):
        raise RuntimeError(
            "Expected model to return logits without a cache."
        )

    vocab_size: int = logits.shape[-1]

    loss: torch.Tensor = F.cross_entropy(
        logits.reshape(
            -1,
            vocab_size,
        ),
        targets.reshape(-1),
    )

    loss.backward()

    gradient_norm: float | None = None

    if max_grad_norm is not None:
        norm: torch.Tensor = (
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm,
            )
        )

        gradient_norm = norm.item()

    optimizer.step()

    return TrainingStepResult(
        loss=loss.item(),
        gradient_norm=gradient_norm,
    )


__all__ = [
    "TrainingStepResult",
    "language_model_training_step",
]
