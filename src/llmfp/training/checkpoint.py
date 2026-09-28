"""Checkpoint helpers for reproducible training experiments."""

from pathlib import Path

import torch
import torch.nn as nn


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_processed: int,
    validation_loss: float | None = None,
) -> None:
    """Save model, optimizer, and scalar training state.

    The checkpoint intentionally stores only tensors and primitive scalar
    values so that it can be loaded with PyTorch weights_only mode.

    Args:
        path: Destination checkpoint path.
        model: Model whose parameters should be saved.
        optimizer: Optimizer whose state should be saved.
        step: Completed optimization step.
        tokens_processed: Number of next-token targets processed so far.
        validation_loss: Optional validation loss associated with the state.
    """
    if step < 0:
        raise ValueError(
            "step must be non-negative."
        )

    if tokens_processed < 0:
        raise ValueError(
            "tokens_processed must be non-negative."
        )

    checkpoint_path = Path(path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "step": step,
            "tokens_processed": tokens_processed,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: torch.device | str | None = None,
) -> dict[str, int | float | None]:
    """Restore model and optional optimizer state from a checkpoint.

    Args:
        path: Checkpoint file to load.
        model: Model instance receiving the saved parameter state.
        optimizer: Optional optimizer receiving the saved optimizer state.
        map_location: Optional PyTorch device mapping used while loading.

    Returns:
        Scalar training metadata: step, tokens_processed, and validation_loss.
    """
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    return {
        "step": int(
            checkpoint["step"]
        ),
        "tokens_processed": int(
            checkpoint["tokens_processed"]
        ),
        "validation_loss": (
            None
            if checkpoint["validation_loss"] is None
            else float(
                checkpoint["validation_loss"]
            )
        ),
    }


__all__ = [
    "load_training_checkpoint",
    "save_training_checkpoint",
]
