"""Utilities for inspecting model structure and parameter counts."""

import torch.nn as nn


def count_parameters(
    module: nn.Module,
    *,
    trainable_only: bool = True,
) -> int:
    """Count scalar parameters in a PyTorch module.

    Args:
        module: Module whose parameters should be counted.
        trainable_only: If true, count only parameters with
            requires_grad=True.

    Returns:
        Number of scalar parameters matching the requested filter.
    """
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def parameter_breakdown(
    module: nn.Module,
    *,
    trainable_only: bool = True,
) -> dict[str, int]:
    """Count parameters for each direct child module.

    Args:
        module: Parent module whose direct children should be inspected.
        trainable_only: If true, count only trainable parameters.

    Returns:
        Mapping from child-module names to scalar parameter counts.
    """
    return {
        name: count_parameters(
            child,
            trainable_only=trainable_only,
        )
        for name, child in module.named_children()
    }


def print_parameter_breakdown(
    module: nn.Module,
    *,
    trainable_only: bool = True,
) -> None:
    """Print parameter counts for direct child modules and the total.

    Args:
        module: Parent module to inspect.
        trainable_only: If true, count only trainable parameters.
    """
    breakdown: dict[str, int] = parameter_breakdown(
        module,
        trainable_only=trainable_only,
    )
    total: int = count_parameters(
        module,
        trainable_only=trainable_only,
    )

    for name, parameter_count in breakdown.items():
        percentage: float = (
            100.0 * parameter_count / total
            if total > 0
            else 0.0
        )
        print(
            f"{name:20s}"
            f"{parameter_count:10d} "
            f"({percentage:5.1f}%)"
        )

    print("-" * 42)
    print(f"{'total':20s}{total:10d}")
