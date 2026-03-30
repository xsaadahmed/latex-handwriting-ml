from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn


def initialize_weights(model: nn.Module) -> None:
    """
    Initialize model weights using He/Kaiming or Xavier depending on layer type.

    - Conv/ConvTranspose: Kaiming normal (good default for ReLU/LeakyReLU nets)
    - Linear: Xavier uniform
    - BatchNorm: weight=1, bias=0
    """

    def init_fn(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    model.apply(init_fn)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in a model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str | Path,
) -> None:
    """
    Save a training checkpoint containing model + optimizer state.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "epoch": int(epoch),
        "loss": float(loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(payload, path)


def load_checkpoint(
    filepath: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str | torch.device] = None,
) -> Tuple[int, float]:
    """
    Load a checkpoint into the model (and optimizer, if provided).

    Returns
    -------
    epoch, loss
        Epoch and loss stored in the checkpoint.
    """
    payload = torch.load(filepath, map_location=map_location)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    epoch = int(payload.get("epoch", 0))
    loss = float(payload.get("loss", 0.0))
    return epoch, loss

