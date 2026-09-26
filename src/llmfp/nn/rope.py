"""Rotary Position Embedding utilities.

The mathematical ideas are derived explicitly in the positional-information
lesson. This module contains the compact reusable implementation used by later
lessons.
"""

import torch


def rope_frequencies(
    head_dim: int,
    base: float = 10000.0,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Compute the angular frequencies used by Rotary Position Embeddings.

    Each adjacent pair of attention-head features shares one frequency.

    Args:
        head_dim: Feature dimension of one attention head.
        base: Base controlling the geometric spacing of RoPE frequencies.
        device: Device on which to create the frequency tensor.

    Returns:
        A tensor of shape (head_dim / 2,) containing one frequency for
        each adjacent feature pair.

    Raises:
        ValueError: If head_dim is not positive and even.
    """
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer for RoPE.")

    # One frequency is shared by each adjacent feature pair:
    # (0, 1), (2, 3), (4, 5), ...
    dimension_indices: torch.Tensor = torch.arange(
        0,
        head_dim,
        2,
        dtype=torch.float32,
        device=device,
    )

    return torch.pow(
        base,
        -dimension_indices / head_dim,
    )


def build_rope_cos_sin(
    sequence_length: int,
    head_dim: int,
    *,
    position_offset: int = 0,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cosine and sine values for a contiguous range of RoPE positions.

    Args:
        sequence_length: Number of token positions to encode.
        head_dim: Feature dimension of one attention head.
        position_offset: Absolute position of the first token. During cached
            decoding, this is typically the number of tokens already stored in
            the KV cache.
        base: Base controlling the geometric spacing of RoPE frequencies.
        device: Device on which to create the tensors.
        dtype: Optional dtype for the returned cosine and sine tensors.

    Returns:
        A pair (cos_values, sin_values), each with shape
        (sequence_length, head_dim / 2).

    Raises:
        ValueError: If sequence_length or position_offset is negative.
    """
    if sequence_length < 0:
        raise ValueError("sequence_length must be non-negative.")

    if position_offset < 0:
        raise ValueError("position_offset must be non-negative.")

    frequencies: torch.Tensor = rope_frequencies(
        head_dim=head_dim,
        base=base,
        device=device,
    )

    # Cached decoding may start at a non-zero absolute position.
    positions: torch.Tensor = torch.arange(
        position_offset,
        position_offset + sequence_length,
        dtype=torch.float32,
        device=device,
    )

    # Broadcasting creates one rotation angle for every
    # (position, frequency) pair.
    angles: torch.Tensor = (
        positions.unsqueeze(1)
        * frequencies.unsqueeze(0)
    )

    cos_values: torch.Tensor = torch.cos(angles)
    sin_values: torch.Tensor = torch.sin(angles)

    if dtype is not None:
        cos_values = cos_values.to(dtype=dtype)
        sin_values = sin_values.to(dtype=dtype)

    return cos_values, sin_values


def apply_rope(
    x: torch.Tensor,
    cos_values: torch.Tensor,
    sin_values: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to an attention tensor.

    Adjacent features in the final dimension are interpreted as independent
    two-dimensional vectors and rotated according to token position.

    Args:
        x: Attention tensor with shape (B, H, T, D).
        cos_values: Cosine values with shape (T, D / 2).
        sin_values: Sine values with shape (T, D / 2).

    Returns:
        Rotated tensor with the same shape (B, H, T, D).

    Raises:
        ValueError: If tensor ranks or RoPE shapes are incompatible.
    """
    if x.ndim != 4:
        raise ValueError("x must have shape (B, H, T, D).")

    sequence_length: int = x.shape[-2]
    head_dim: int = x.shape[-1]

    if head_dim % 2 != 0:
        raise ValueError("The head dimension must be even for RoPE.")

    expected_shape: tuple[int, int] = (
        sequence_length,
        head_dim // 2,
    )

    if tuple(cos_values.shape) != expected_shape:
        raise ValueError(
            "cos_values must have shape "
            f"{expected_shape}, got {tuple(cos_values.shape)}."
        )

    if tuple(sin_values.shape) != expected_shape:
        raise ValueError(
            "sin_values must have shape "
            f"{expected_shape}, got {tuple(sin_values.shape)}."
        )

    # Split adjacent feature pairs into their first and second components.
    x_even: torch.Tensor = x[..., 0::2]
    x_odd: torch.Tensor = x[..., 1::2]

    # Add batch and head axes so the same positional rotations broadcast
    # across all examples and attention heads.
    cos_values = cos_values.unsqueeze(0).unsqueeze(0)
    sin_values = sin_values.unsqueeze(0).unsqueeze(0)

    # Apply the standard 2D rotation formula independently to every pair.
    rotated_even: torch.Tensor = (
        x_even * cos_values
        - x_odd * sin_values
    )
    rotated_odd: torch.Tensor = (
        x_even * sin_values
        + x_odd * cos_values
    )

    # Reconstruct the adjacent pairs:
    # [x0', x1', x2', x3', ...]
    rotated_pairs: torch.Tensor = torch.stack(
        [rotated_even, rotated_odd],
        dim=-1,
    )

    return rotated_pairs.flatten(start_dim=-2)
