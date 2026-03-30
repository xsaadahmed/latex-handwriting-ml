from __future__ import annotations

from typing import Any

import torch
from torch import nn


class StyleEncoder(nn.Module):
    """
    Placeholder style encoder.

    Encodes handwriting samples (or writer IDs) into a continuous style embedding
    that conditions the generator. Replace with a more appropriate architecture
    for your dataset.
    """

    def __init__(self, image_channels: int = 1, style_dim: int = 64) -> None:
        super().__init__()
        self.image_channels = image_channels
        self.style_dim = style_dim

        self.net = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(256, style_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Encode an image into a style embedding."""
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def build_style_encoder(config: dict[str, Any]) -> StyleEncoder:
    """Factory to build a style encoder from a config dictionary."""
    image_channels = int(config.get("image_channels", 1))
    style_dim = int(config.get("style_dim", 64))
    return StyleEncoder(image_channels=image_channels, style_dim=style_dim)

