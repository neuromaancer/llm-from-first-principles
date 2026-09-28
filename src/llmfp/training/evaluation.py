"""Evaluation helpers for autoregressive language models."""

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from llmfp.data import get_batch


@torch.no_grad()
def estimate_language_model_loss(
    model: nn.Module,
    token_splits: Mapping[str, torch.Tensor],
    *,
    batch_size: int,
    context_length: int,
    evaluation_batches: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Estimate mean next-token cross-entropy on one or more token streams.

    Args:
        model: Language model returning logits with shape (B, T, V).
        token_splits: Named one-dimensional token streams.
        batch_size: Number of sampled sequences per evaluation batch.
        context_length: Number of prediction positions per sequence.
        evaluation_batches: Number of random batches evaluated per split.
        device: Device on which model evaluation runs.
        generator: Optional random generator for reproducible window sampling.

    Returns:
        Mapping from split name to mean cross-entropy loss.

    Raises:
        ValueError: If no splits are supplied or evaluation_batches is invalid.
        RuntimeError: If the model unexpectedly returns a cache tuple.
    """
    if not token_splits:
        raise ValueError(
            "token_splits must contain at least one split."
        )

    if evaluation_batches <= 0:
        raise ValueError(
            "evaluation_batches must be positive."
        )

    was_training: bool = model.training
    model.eval()

    results: dict[str, float] = {}

    try:
        for split_name, tokens in token_splits.items():
            total_loss: float = 0.0

            for _ in range(evaluation_batches):
                inputs, targets = get_batch(
                    tokens,
                    batch_size=batch_size,
                    context_length=context_length,
                    device=device,
                    generator=generator,
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

                total_loss += loss.item()

            results[split_name] = (
                total_loss / evaluation_batches
            )
    finally:
        if was_training:
            model.train()

    return results


__all__ = [
    "estimate_language_model_loss",
]
