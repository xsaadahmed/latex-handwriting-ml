from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch import optim

from .discriminator import PatchGANDiscriminator
from .generator import UNetGenerator
from .utils import count_parameters, load_checkpoint, save_checkpoint


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_shape(t: torch.Tensor, expected: Tuple[int, ...], name: str) -> None:
    if tuple(t.shape) != expected:
        raise AssertionError(f"{name} shape mismatch: got {tuple(t.shape)}, expected {expected}")


def run_tests() -> None:
    device = _device()
    print(f"Using device: {device}")

    batch_size = 4
    h = w = 256

    gen = UNetGenerator(in_channels=1, out_channels=1).to(device)
    disc = PatchGANDiscriminator(in_channels=2).to(device)

    print(f"Generator params: {count_parameters(gen):,}")
    print(f"Discriminator params: {count_parameters(disc):,}")

    printed = torch.randn(batch_size, 1, h, w, device=device)
    real_hw = torch.randn(batch_size, 1, h, w, device=device)

    # Forward pass: generator
    with torch.no_grad():
        fake_hw = gen(printed)
    _assert_shape(fake_hw, (batch_size, 1, h, w), "Generator output")
    print(f"Generator output shape: {tuple(fake_hw.shape)}")

    # Verify skip connections were exercised (shape compatibility checks from forward pass)
    skips = gen.get_last_skip_shapes()
    if len(skips) != 7:
        raise AssertionError(f"Expected 7 skip concatenations, got {len(skips)}")
    for idx, (up_shape, enc_shape) in enumerate(skips, start=1):
        if up_shape[0] != enc_shape[0] or up_shape[2:] != enc_shape[2:]:
            raise AssertionError(
                f"Skip {idx} incompatible: up={up_shape}, enc={enc_shape} (batch/spatial should match)"
            )
    print("U-Net skip connection checks: OK")

    # Forward pass: discriminator (patch output, not scalar)
    with torch.no_grad():
        real_pred = disc(printed, real_hw)
        fake_pred = disc(printed, fake_hw)
    print(f"Discriminator real patch shape: {tuple(real_pred.shape)}")
    print(f"Discriminator fake patch shape: {tuple(fake_pred.shape)}")
    if real_pred.ndim != 4 or real_pred.shape[1] != 1:
        raise AssertionError("Discriminator output must be a (B,1,H',W') patch tensor.")

    # Checkpoint save/load smoke test (generator)
    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints" / "tests"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "gen_test_checkpoint.pt"

    opt = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    save_checkpoint(gen, opt, epoch=1, loss=0.123, filepath=ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}")

    epoch, loss = load_checkpoint(ckpt_path, gen, optimizer=opt, map_location=device)
    print(f"Loaded checkpoint epoch={epoch}, loss={loss}")

    print("All Pix2Pix model tests passed.")


if __name__ == "__main__":
    run_tests()

