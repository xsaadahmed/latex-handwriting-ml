"""
Data loading utilities for the LaTeX Handwriting ML project.

This package provides:
- `PairedEquationDataset` for paired printed/handwritten equations.
- Helpers for synthetic dataset creation and DataLoader construction.
"""

from .dataset import PairedEquationDataset, create_synthetic_dataset, get_dataloaders

__all__ = ["PairedEquationDataset", "create_synthetic_dataset", "get_dataloaders"]

