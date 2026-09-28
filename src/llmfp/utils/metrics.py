"""Small evaluation metrics used by training experiments."""

import math


def perplexity_from_loss(
    loss: float,
) -> float:
    """Convert mean natural-log cross-entropy into perplexity.

    Args:
        loss: Mean cross-entropy measured with natural logarithms.

    Returns:
        Perplexity equal to `exp(loss)`.
    """
    return math.exp(loss)


__all__ = [
    "perplexity_from_loss",
]
