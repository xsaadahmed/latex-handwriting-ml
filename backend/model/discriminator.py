from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class PatchGANDiscriminator(nn.Module):
    """
    Pix2Pix PatchGAN discriminator (70x70-style patch discriminator).

    Takes a source image (printed) and target image (real or generated handwritten),
    concatenates them along the channel dimension, and outputs a patch-wise real/fake
    prediction map (logits).

    - Input:  source (B, 1, 256, 256), target (B, 1, 256, 256) -> concat (B, 2, 256, 256)
    - Output: patch logits tensor (B, 1, H', W') (not a single scalar)
    """

    def __init__(self, in_channels: int = 2) -> None:
        super().__init__()
        self.in_channels = in_channels

        def conv_block(
            cin: int,
            cout: int,
            stride: int,
            use_batchnorm: bool,
        ) -> nn.Sequential:
            layers: List[nn.Module] = [
                nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1, bias=not use_batchnorm)
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        # 5 conv layers: 64 -> 128 -> 256 -> 512 -> 1
        # Stride=2 for first 4 layers, stride=1 for the last layer (per spec).
        self.layer1 = conv_block(in_channels, 64, stride=2, use_batchnorm=False)
        self.layer2 = conv_block(64, 128, stride=2, use_batchnorm=True)
        self.layer3 = conv_block(128, 256, stride=2, use_batchnorm=True)
        self.layer4 = conv_block(256, 512, stride=2, use_batchnorm=True)
        self.layer5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # patch logits

        self.apply(self._init_weights_normal)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        source:
            Source/conditioning image (printed), shape ``(B, 1, 256, 256)``.
        target:
            Target image (real handwritten or generated), shape ``(B, 1, 256, 256)``.

        Returns
        -------
        torch.Tensor
            Patch-wise logits of shape ``(B, 1, H', W')``.
        """
        if source.shape[0] != target.shape[0]:
            raise ValueError("Source and target must have the same batch size.")
        if source.shape[2:] != target.shape[2:]:
            raise ValueError("Source and target must have matching spatial dimensions.")

        x = torch.cat([source, target], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    @staticmethod
    def _init_weights_normal(m: nn.Module) -> None:
        """
        Initialize weights with N(0, 0.02) as commonly used in Pix2Pix/CycleGAN.
        """
        classname = m.__class__.__name__
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def save_weights(self, filepath: str | Path) -> None:
        """Save model weights to a file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, filepath: str | Path, map_location: str | torch.device | None = None) -> None:
        """Load model weights from a file."""
        state = torch.load(filepath, map_location=map_location)
        self.load_state_dict(state)

    def print_summary(
        self,
        input_size: Tuple[int, int, int, int] = (1, 1, 256, 256),
        device: Optional[torch.device] = None,
    ) -> None:
        """Print a minimal summary of patch output shapes."""
        dev = device or next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            src = torch.randn(*input_size, device=dev)
            tgt = torch.randn(*input_size, device=dev)
            out = self(src, tgt)
        print("PatchGANDiscriminator Summary")
        print(f"Source/Target input: {input_size} + {input_size} -> concat channels={self.in_channels}")
        print(f"Output patches: {tuple(out.shape)}")
        print(f"Trainable parameters: {self.count_parameters(True):,}")


def build_discriminator(config: Dict[str, object]) -> PatchGANDiscriminator:
    """
    Factory to build a PatchGAN discriminator from a config dictionary.

    Expected keys (optional):
    - in_channels (default 2)
    """
    in_channels = int(config.get("in_channels", 2))  # type: ignore[arg-type]
    return PatchGANDiscriminator(in_channels=in_channels)


# Backwards-compatible alias.
Discriminator = PatchGANDiscriminator

