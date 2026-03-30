from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .config import ModelConfig, load_config
from .generator import build_generator
from .style_encoder import build_style_encoder


def load_models_for_inference(
    config: Dict[str, Any],
    generator_checkpoint: str | Path | None = None,
    style_encoder_checkpoint: str | Path | None = None,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load generator and style encoder for inference.

    This function is intentionally lightweight and can be expanded with
    additional logic (e.g. EMA weights, different checkpoints per style).
    """
    model_cfg = ModelConfig.from_dict(config)

    gen = build_generator(
        {
            "latent_dim": model_cfg.latent_dim,
            "style_dim": model_cfg.style_dim,
            "image_channels": model_cfg.image_channels,
        }
    )
    style_enc = build_style_encoder(
        {"image_channels": model_cfg.image_channels, "style_dim": model_cfg.style_dim}
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen.to(device)
    style_enc.to(device)

    if generator_checkpoint is not None:
        state = torch.load(generator_checkpoint, map_location=device)
        gen.load_state_dict(state)

    if style_encoder_checkpoint is not None:
        state = torch.load(style_encoder_checkpoint, map_location=device)
        style_enc.load_state_dict(state)

    gen.eval()
    style_enc.eval()
    return gen, style_enc


def load_from_yaml(
    config_path: str | Path,
    generator_checkpoint: str | Path | None = None,
    style_encoder_checkpoint: str | Path | None = None,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Convenience wrapper to load config and then models for inference."""
    cfg = load_config(config_path)
    return load_models_for_inference(cfg, generator_checkpoint, style_encoder_checkpoint)

