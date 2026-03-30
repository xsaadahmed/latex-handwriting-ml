from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set random seed for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the default computation device."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def is_debug_mode() -> bool:
    """Return True if running in debug/development mode."""
    return os.getenv("DEBUG", "0") == "1"

