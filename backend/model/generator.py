from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


class UNetDown(nn.Module):
    """
    U-Net encoder (downsampling) block used in Pix2Pix.

    Conv2d(k=4,s=2,p=1) -> (optional) BatchNorm2d -> LeakyReLU(0.2).
    """

    def __init__(self, in_channels: int, out_channels: int, use_batchnorm: bool) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not use_batchnorm)
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.block(x)


class UNetUp(nn.Module):
    """
    U-Net decoder (upsampling) block used in Pix2Pix.

    ConvTranspose2d(k=4,s=2,p=1) -> BatchNorm2d -> ReLU -> (optional) Dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, use_dropout: bool) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.block(x)


@dataclass
class SummaryRow:
    name: str
    in_shape: Tuple[int, ...]
    out_shape: Tuple[int, ...]


class UNetGenerator(nn.Module):
    """
    Pix2Pix U-Net Generator (256x256 input).

    - Input:  (B, 1, 256, 256)
    - Output: (B, 1, 256, 256) with Tanh activation in [-1, 1].
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder (8 downsampling blocks).
        # Channel progression: 64 -> 128 -> 256 -> 512 -> 512 -> 512 -> 512 -> 512
        self.down1 = UNetDown(in_channels, 64, use_batchnorm=False)
        self.down2 = UNetDown(64, 128, use_batchnorm=True)
        self.down3 = UNetDown(128, 256, use_batchnorm=True)
        self.down4 = UNetDown(256, 512, use_batchnorm=True)
        self.down5 = UNetDown(512, 512, use_batchnorm=True)
        self.down6 = UNetDown(512, 512, use_batchnorm=True)
        self.down7 = UNetDown(512, 512, use_batchnorm=True)
        self.down8 = UNetDown(512, 512, use_batchnorm=True)

        # Decoder (8 upsampling blocks).
        # Use dropout (0.5) in first 3 up blocks.
        self.up1 = UNetUp(512, 512, use_dropout=True)  # d8 -> u1, then concat with d7
        self.up2 = UNetUp(1024, 512, use_dropout=True)
        self.up3 = UNetUp(1024, 512, use_dropout=True)
        self.up4 = UNetUp(1024, 512, use_dropout=False)
        self.up5 = UNetUp(1024, 256, use_dropout=False)
        self.up6 = UNetUp(512, 128, use_dropout=False)
        self.up7 = UNetUp(256, 64, use_dropout=False)

        # Final layer (no BatchNorm), mirror Pix2Pix: ReLU -> ConvT -> Tanh
        self.final = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self._last_skip_shapes: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass with U-Net skip connections.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, 1, 256, 256)`` (printed equation image).
        """
        self._last_skip_shapes = []

        d1 = self.down1(x)   # (B, 64, 128, 128)
        d2 = self.down2(d1)  # (B, 128, 64, 64)
        d3 = self.down3(d2)  # (B, 256, 32, 32)
        d4 = self.down4(d3)  # (B, 512, 16, 16)
        d5 = self.down5(d4)  # (B, 512, 8, 8)
        d6 = self.down6(d5)  # (B, 512, 4, 4)
        d7 = self.down7(d6)  # (B, 512, 2, 2)
        d8 = self.down8(d7)  # (B, 512, 1, 1)

        u1 = self.up1(d8)  # (B, 512, 2, 2)
        self._last_skip_shapes.append((tuple(u1.shape), tuple(d7.shape)))
        u1 = torch.cat([u1, d7], dim=1)  # (B, 1024, 2, 2)

        u2 = self.up2(u1)  # (B, 512, 4, 4)
        self._last_skip_shapes.append((tuple(u2.shape), tuple(d6.shape)))
        u2 = torch.cat([u2, d6], dim=1)  # (B, 1024, 4, 4)

        u3 = self.up3(u2)  # (B, 512, 8, 8)
        self._last_skip_shapes.append((tuple(u3.shape), tuple(d5.shape)))
        u3 = torch.cat([u3, d5], dim=1)  # (B, 1024, 8, 8)

        u4 = self.up4(u3)  # (B, 512, 16, 16)
        self._last_skip_shapes.append((tuple(u4.shape), tuple(d4.shape)))
        u4 = torch.cat([u4, d4], dim=1)  # (B, 1024, 16, 16)

        u5 = self.up5(u4)  # (B, 256, 32, 32)
        self._last_skip_shapes.append((tuple(u5.shape), tuple(d3.shape)))
        u5 = torch.cat([u5, d3], dim=1)  # (B, 512, 32, 32)

        u6 = self.up6(u5)  # (B, 128, 64, 64)
        self._last_skip_shapes.append((tuple(u6.shape), tuple(d2.shape)))
        u6 = torch.cat([u6, d2], dim=1)  # (B, 256, 64, 64)

        u7 = self.up7(u6)  # (B, 64, 128, 128)
        self._last_skip_shapes.append((tuple(u7.shape), tuple(d1.shape)))
        u7 = torch.cat([u7, d1], dim=1)  # (B, 128, 128, 128)

        out = self.final(u7)  # (B, 1, 256, 256)
        return out

    def get_last_skip_shapes(self) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """
        Return shapes recorded during the most recent forward pass.

        Each entry is ``(up_block_output_shape, encoder_skip_shape)``.
        """
        return list(self._last_skip_shapes)

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
        """
        Print a minimal architecture summary with input/output shapes per module.

        This avoids extra dependencies (e.g., torchinfo) by using forward hooks.
        """
        dev = device or next(self.parameters()).device
        rows: List[SummaryRow] = []
        hooks: List[torch.utils.hooks.RemovableHandle] = []

        def _hook(name: str):
            def fn(module: nn.Module, inp: Tuple[torch.Tensor, ...], out: torch.Tensor):
                if not inp:
                    return
                in_shape = tuple(inp[0].shape)
                out_shape = tuple(out.shape) if isinstance(out, torch.Tensor) else tuple()
                rows.append(SummaryRow(name=name, in_shape=in_shape, out_shape=out_shape))

            return fn

        for name, module in self.named_modules():
            if name and not list(module.children()):
                hooks.append(module.register_forward_hook(_hook(name)))

        self.eval()
        with torch.no_grad():
            dummy = torch.randn(*input_size, device=dev)
            _ = self(dummy)

        for h in hooks:
            h.remove()

        print("UNetGenerator Summary")
        print(f"Input: {input_size}")
        for r in rows:
            print(f"- {r.name}: {r.in_shape} -> {r.out_shape}")
        print(f"Trainable parameters: {self.count_parameters(True):,}")


def build_generator(config: Dict[str, object]) -> UNetGenerator:
    """
    Factory to build a Pix2Pix U-Net generator from a config dictionary.

    Expected keys (optional):
    - in_channels
    - out_channels
    """
    in_channels = int(config.get("in_channels", 1))  # type: ignore[arg-type]
    out_channels = int(config.get("out_channels", 1))  # type: ignore[arg-type]
    return UNetGenerator(in_channels=in_channels, out_channels=out_channels)


# Backwards-compatible alias.
Generator = UNetGenerator

