"""Utilities for turning token streams into language-model batches."""

import torch


def split_token_stream(
    tokens: torch.Tensor,
    train_fraction: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split one ordered token stream into contiguous train/validation parts.

    Splitting before sampling windows prevents overlapping windows from being
    assigned across the train/validation boundary.

    Args:
        tokens: One-dimensional integer token stream.
        train_fraction: Fraction of tokens assigned to training.

    Returns:
        Tuple `(train_tokens, validation_tokens)`.

    Raises:
        ValueError: If the token stream or fraction is invalid.
    """
    if tokens.ndim != 1:
        raise ValueError(
            "tokens must be a one-dimensional token stream."
        )

    if len(tokens) < 2:
        raise ValueError(
            "tokens must contain at least two elements."
        )

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            "train_fraction must be strictly between 0 and 1."
        )

    split_index: int = int(
        train_fraction * len(tokens)
    )

    if split_index == 0 or split_index == len(tokens):
        raise ValueError(
            "train_fraction leaves one split empty."
        )

    return (
        tokens[:split_index],
        tokens[split_index:],
    )


def get_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random next-token training windows from a token stream.

    Each sampled source window contains `context_length + 1` tokens. The
    first `context_length` positions become model inputs, and the same window
    shifted by one token becomes the targets.

    Args:
        tokens: One-dimensional integer token stream.
        batch_size: Number of independent sequences in the batch.
        context_length: Number of prediction positions per sequence.
        device: Optional destination device for the returned tensors.
        generator: Optional random generator for reproducible sampling.

    Returns:
        Tuple `(inputs, targets)`, each with shape `(B, T)`.

    Raises:
        ValueError: If arguments cannot define at least one training window.
    """
    if tokens.ndim != 1:
        raise ValueError(
            "tokens must be a one-dimensional token stream."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if context_length <= 0:
        raise ValueError(
            "context_length must be positive."
        )

    if len(tokens) <= context_length:
        raise ValueError(
            "token stream must contain at least context_length + 1 tokens."
        )

    maximum_start: int = (
        len(tokens) - context_length - 1
    )

    start_indices: torch.Tensor = torch.randint(
        low=0,
        high=maximum_start + 1,
        size=(batch_size,),
        generator=generator,
        device=tokens.device,
    )

    offsets: torch.Tensor = torch.arange(
        context_length,
        device=tokens.device,
    )

    positions: torch.Tensor = (
        start_indices.unsqueeze(1)
        + offsets.unsqueeze(0)
    )

    inputs: torch.Tensor = tokens[
        positions
    ]

    targets: torch.Tensor = tokens[
        positions + 1
    ]

    if device is not None:
        inputs = inputs.to(device)
        targets = targets.to(device)

    return (
        inputs,
        targets,
    )


def tokens_per_step(
    batch_size: int,
    context_length: int,
) -> int:
    """Return the number of next-token targets in one training batch."""
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    if context_length <= 0:
        raise ValueError(
            "context_length must be positive."
        )

    return batch_size * context_length


__all__ = [
    "get_batch",
    "split_token_stream",
    "tokens_per_step",
]
