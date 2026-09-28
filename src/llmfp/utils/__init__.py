"""Small inspection, metric, and development helpers."""

from .inspection import (
    count_parameters,
    parameter_breakdown,
    print_parameter_breakdown,
)
from .metrics import perplexity_from_loss

__all__ = [
    "count_parameters",
    "parameter_breakdown",
    "perplexity_from_loss",
    "print_parameter_breakdown",
]
