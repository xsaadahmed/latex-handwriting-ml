from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    """Typed view over the subset of config relevant to models."""

    latent_dim: int = 128
    style_dim: int = 64
    image_channels: int = 1

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        model_cfg = data.get("model", {}) or {}
        image_cfg = data.get("image", {}) or {}
        return cls(
            latent_dim=int(model_cfg.get("latent_dim", 128)),
            style_dim=int(model_cfg.get("style_dim", 64)),
            image_channels=int(image_cfg.get("channels", 1)),
        )


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config file into a plain dictionary."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

