"""Sampling filters and next-token selection helpers."""

import torch
import torch.nn.functional as F


def top_k_filter(
    logits: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Keep only the k largest logits along the vocabulary dimension.

    Args:
        logits: Logits with shape `(..., V)`.
        k: Number of highest-logit candidates to retain.

    Returns:
        Filtered logits with the same shape. Removed candidates are replaced
        by negative infinity.
    """
    vocab_size: int = logits.shape[-1]

    if not 1 <= k <= vocab_size:
        raise ValueError(
            "k must be between 1 and the vocabulary size."
        )

    top_values, _ = torch.topk(
        logits,
        k=k,
        dim=-1,
    )

    threshold: torch.Tensor = (
        top_values[..., -1:]
    )

    return logits.masked_fill(
        logits < threshold,
        float("-inf"),
    )


def top_p_filter(
    logits: torch.Tensor,
    p: float,
) -> torch.Tensor:
    """Keep the smallest high-probability candidate set covering mass p.

    Args:
        logits: Logits with shape `(..., V)`.
        p: Cumulative probability threshold in `(0, 1]`.

    Returns:
        Filtered logits with the same shape and original vocabulary order.
    """
    if not 0.0 < p <= 1.0:
        raise ValueError(
            "p must be greater than 0 and at most 1."
        )

    sorted_logits, sorted_indices = torch.sort(
        logits,
        dim=-1,
        descending=True,
    )

    sorted_probabilities: torch.Tensor = F.softmax(
        sorted_logits,
        dim=-1,
    )

    cumulative_probabilities: torch.Tensor = torch.cumsum(
        sorted_probabilities,
        dim=-1,
    )

    sorted_remove_mask: torch.Tensor = (
        cumulative_probabilities > p
    )

    sorted_remove_mask[..., 1:] = (
        sorted_remove_mask[..., :-1].clone()
    )
    sorted_remove_mask[..., 0] = False

    filtered_sorted_logits: torch.Tensor = (
        sorted_logits.masked_fill(
            sorted_remove_mask,
            float("-inf"),
        )
    )

    filtered_logits: torch.Tensor = torch.full_like(
        logits,
        float("-inf"),
    )

    filtered_logits.scatter_(
        dim=-1,
        index=sorted_indices,
        src=filtered_sorted_logits,
    )

    return filtered_logits


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token from next-token logits.

    Temperature is applied before top-k/top-p filtering. If both filters are
    supplied, top-k is applied first and top-p second.

    Args:
        logits: Next-token logits with shape `(B, V)`.
        temperature: Positive temperature scaling factor.
        top_k: Optional number of highest-logit candidates to retain.
        top_p: Optional nucleus probability threshold.
        generator: Optional random generator for reproducible sampling.

    Returns:
        Sampled token IDs with shape `(B, 1)`.
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape (B, V)."
        )

    if temperature <= 0.0:
        raise ValueError(
            "temperature must be positive."
        )

    filtered_logits: torch.Tensor = (
        logits / temperature
    )

    if top_k is not None:
        filtered_logits = top_k_filter(
            filtered_logits,
            k=top_k,
        )

    if top_p is not None:
        filtered_logits = top_p_filter(
            filtered_logits,
            p=top_p,
        )

    probabilities: torch.Tensor = F.softmax(
        filtered_logits,
        dim=-1,
    )

    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )


__all__ = [
    "sample_next_token",
    "top_k_filter",
    "top_p_filter",
]
