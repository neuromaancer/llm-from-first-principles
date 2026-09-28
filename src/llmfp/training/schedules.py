"""Learning-rate schedules reused by training experiments."""

import math


def linear_warmup(
    step: int,
    warmup_steps: int,
    peak_learning_rate: float,
) -> float:
    """Linearly increase the learning rate during warmup.

    Args:
        step: One-based optimization step inside the warmup interval.
        warmup_steps: Number of warmup steps.
        peak_learning_rate: Learning rate reached at the end of warmup.

    Returns:
        Learning rate for the requested warmup step.
    """
    if warmup_steps <= 0:
        raise ValueError(
            "warmup_steps must be positive."
        )

    if not 1 <= step <= warmup_steps:
        raise ValueError(
            "step must be inside the warmup interval."
        )

    if peak_learning_rate < 0.0:
        raise ValueError(
            "peak_learning_rate must be non-negative."
        )

    progress: float = (
        step / warmup_steps
    )

    return (
        peak_learning_rate * progress
    )


def cosine_decay(
    progress: float,
    peak_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    """Cosine-decay a learning rate from peak to minimum.

    Args:
        progress: Decay progress in the closed interval [0, 1].
        peak_learning_rate: Learning rate at progress 0.
        minimum_learning_rate: Learning rate at progress 1.

    Returns:
        Cosine-decayed learning rate.
    """
    if not 0.0 <= progress <= 1.0:
        raise ValueError(
            "progress must be between 0 and 1."
        )

    if peak_learning_rate < 0.0:
        raise ValueError(
            "peak_learning_rate must be non-negative."
        )

    if minimum_learning_rate < 0.0:
        raise ValueError(
            "minimum_learning_rate must be non-negative."
        )

    if minimum_learning_rate > peak_learning_rate:
        raise ValueError(
            "minimum_learning_rate must not exceed peak_learning_rate."
        )

    cosine_factor: float = 0.5 * (
        1.0
        + math.cos(
            math.pi * progress
        )
    )

    return (
        minimum_learning_rate
        + (
            peak_learning_rate
            - minimum_learning_rate
        )
        * cosine_factor
    )


def warmup_cosine_learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak_learning_rate: float,
    minimum_learning_rate: float,
) -> float:
    """Combine linear warmup with cosine learning-rate decay.

    Args:
        step: One-based optimization step.
        total_steps: Total number of optimization steps.
        warmup_steps: Number of initial linear-warmup steps.
        peak_learning_rate: Maximum learning rate.
        minimum_learning_rate: Final learning rate.

    Returns:
        Learning rate for the requested optimization step.
    """
    if not 1 <= step <= total_steps:
        raise ValueError(
            "step must be between 1 and total_steps."
        )

    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            "warmup_steps must be in [0, total_steps)."
        )

    if step <= warmup_steps:
        return linear_warmup(
            step=step,
            warmup_steps=warmup_steps,
            peak_learning_rate=peak_learning_rate,
        )

    decay_steps: int = (
        total_steps - warmup_steps
    )

    decay_step: int = (
        step - warmup_steps
    )

    decay_progress: float = (
        decay_step / decay_steps
    )

    return cosine_decay(
        progress=decay_progress,
        peak_learning_rate=peak_learning_rate,
        minimum_learning_rate=minimum_learning_rate,
    )


__all__ = [
    "cosine_decay",
    "linear_warmup",
    "warmup_cosine_learning_rate",
]
