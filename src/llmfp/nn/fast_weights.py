"""Small associative-memory helpers used by fast-weight experiments."""

import torch


def read_fast_weight_memory(
    key: torch.Tensor,
    memory: torch.Tensor,
) -> torch.Tensor:
    """Read one value from a fast-weight memory matrix.

    Args:
        key: Query/key vector with shape `(M)`.
        memory: Memory matrix with shape `(M, D_V)`.

    Returns:
        Predicted value with shape `(D_V)`.

    Raises:
        ValueError: If the input shapes are inconsistent.
    """
    if key.ndim != 1:
        raise ValueError(
            "key must have shape (M)."
        )

    if memory.ndim != 2:
        raise ValueError(
            "memory must have shape (M, D_V)."
        )

    if key.shape[0] != memory.shape[0]:
        raise ValueError(
            "key dimension must match the first memory dimension."
        )

    return key @ memory


def additive_memory_update(
    memory: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Write a full key-value association into fast-weight memory.

    The update is

        W_new = W + k v^T.

    Args:
        memory: Current memory with shape `(M, D_V)`.
        key: Key vector with shape `(M)`.
        value: Value vector with shape `(D_V)`.

    Returns:
        Updated memory with shape `(M, D_V)`.

    Raises:
        ValueError: If the input shapes are inconsistent.
    """
    if key.ndim != 1 or value.ndim != 1:
        raise ValueError(
            "key and value must both be rank-1 tensors."
        )

    if memory.ndim != 2:
        raise ValueError(
            "memory must have shape (M, D_V)."
        )

    if memory.shape != (
        key.shape[0],
        value.shape[0],
    ):
        raise ValueError(
            "memory shape must match key and value dimensions."
        )

    update: torch.Tensor = (
        key.unsqueeze(-1)
        * value.unsqueeze(0)
    )

    return memory + update


def delta_memory_update(
    memory: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    write_rate: float = 1.0,
) -> torch.Tensor:
    """Update fast-weight memory using the Delta Rule.

    The memory first predicts the value associated with the key,

        v_hat = k^T W,

    then writes only the prediction error:

        W_new = W + eta k (v - v_hat)^T.

    Args:
        memory: Current memory with shape `(M, D_V)`.
        key: Key vector with shape `(M)`.
        value: Desired value with shape `(D_V)`.
        write_rate: Scalar memory-update strength.

    Returns:
        Updated memory with shape `(M, D_V)`.

    Raises:
        ValueError: If the input shapes are inconsistent or if
            `write_rate` is negative.
    """
    if write_rate < 0:
        raise ValueError(
            "write_rate must be non-negative."
        )

    prediction: torch.Tensor = read_fast_weight_memory(
        key,
        memory,
    )

    if value.ndim != 1:
        raise ValueError(
            "value must have shape (D_V)."
        )

    if value.shape != prediction.shape:
        raise ValueError(
            "value dimension must match the memory output dimension."
        )

    error: torch.Tensor = (
        value - prediction
    )

    update: torch.Tensor = (
        key.unsqueeze(-1)
        * error.unsqueeze(0)
    )

    return memory + write_rate * update


__all__ = [
    "additive_memory_update",
    "delta_memory_update",
    "read_fast_weight_memory",
]
