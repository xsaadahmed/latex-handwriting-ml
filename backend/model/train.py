from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn, optim

from backend.data import get_dataloaders

from .config import load_config
from .discriminator import build_discriminator
from .generator import build_generator
from .losses import Pix2PixLoss
from .visualize import plot_training_curves, save_comparison_grid


def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validate(
    generator: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion_l1: nn.Module,
    device: torch.device,
    samples_dir: Path,
    epoch: int,
) -> float:
    """
    Run validation over the validation set and compute average L1 loss.

    Also saves a small set of comparison grids for qualitative inspection.
    """
    generator.eval()
    total_l1 = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            printed = batch["printed"].to(device)
            real_hw = batch["handwritten"].to(device)

            fake_hw = generator(printed)

            l1 = criterion_l1(fake_hw, real_hw)
            batch_size = printed.size(0)
            total_l1 += float(l1.item()) * batch_size
            num_samples += batch_size

            # Save only a few validation grids per epoch.
            if batch_idx < 2:
                save_comparison_grid(
                    printed=printed,
                    real_handwritten=real_hw,
                    fake_handwritten=fake_hw,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    save_dir=samples_dir / "val",
                )

    return total_l1 / max(1, num_samples)


def train(config_path: str | Path = "config.yaml") -> None:
    """
    Main Pix2Pix training loop.
    """
    project_root = Path(__file__).resolve().parents[2]
    cfg = load_config(config_path)

    image_cfg = cfg.get("image", {})
    train_cfg = cfg.get("training", {})
    data_cfg = cfg.get("data", {})
    paths_cfg = cfg.get("paths", {})

    epochs = int(train_cfg.get("epochs", 200))
    batch_size = int(train_cfg.get("batch_size", 16))
    lr = float(train_cfg.get("learning_rate", 2e-4))
    beta1 = float(train_cfg.get("beta1", 0.5))
    beta2 = float(train_cfg.get("beta2", 0.999))
    lambda_l1 = float(train_cfg.get("lambda_l1", 100.0))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 10))
    log_interval = int(train_cfg.get("log_interval", 10))
    sample_interval = int(train_cfg.get("sample_interval", 1))

    _image_size = int(data_cfg.get("image_size", image_cfg.get("height", 256)))
    train_split = float(data_cfg.get("train_split", 0.8))
    num_workers = int(data_cfg.get("num_workers", 4))

    checkpoint_dir = project_root / paths_cfg.get("checkpoint_dir", "checkpoints")
    output_dir = project_root / paths_cfg.get("output_dir", "outputs")
    samples_dir = project_root / paths_cfg.get("samples_dir", "outputs/samples")
    curves_path = output_dir / "training_curves.json"
    curves_plot_path = output_dir / "training_curves.png"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        train_split=train_split,
        num_workers=num_workers,
    )

    # Models
    gen = build_generator({"in_channels": 1, "out_channels": 1})
    disc = build_discriminator({"in_channels": 2})
    gen.to(device)
    disc.to(device)

    # Optimizers and losses
    opt_g = optim.Adam(gen.parameters(), lr=lr, betas=(beta1, beta2))
    opt_d = optim.Adam(disc.parameters(), lr=lr, betas=(beta1, beta2))

    pix2pix_loss = Pix2PixLoss(lambda_L1=lambda_l1)
    l1_val_loss = nn.L1Loss()

    history: Dict[str, Any] = {
        "G_total": [],
        "G_adv": [],
        "G_L1": [],
        "D_total": [],
        "val_L1": [],
    }

    for epoch in range(1, epochs + 1):
        gen.train()
        disc.train()

        epoch_g_total = 0.0
        epoch_g_adv = 0.0
        epoch_g_l1 = 0.0
        epoch_d_total = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            printed = batch["printed"].to(device)
            real_hw = batch["handwritten"].to(device)

            # --------------------
            # Train Discriminator
            # --------------------
            opt_d.zero_grad(set_to_none=True)

            with torch.no_grad():
                fake_hw = gen(printed)

            pred_real = disc(printed, real_hw)
            pred_fake = disc(printed, fake_hw)

            loss_d, loss_d_real, loss_d_fake = pix2pix_loss.discriminator_loss(
                real_predictions=pred_real,
                fake_predictions=pred_fake,
            )
            loss_d.backward()
            opt_d.step()

            # ----------------
            # Train Generator
            # ----------------
            opt_g.zero_grad(set_to_none=True)

            fake_hw = gen(printed)
            pred_fake_for_g = disc(printed, fake_hw)

            loss_g_total, loss_g_adv, loss_g_l1 = pix2pix_loss.generator_loss(
                fake_output=fake_hw,
                real_output=real_hw,
                fake_predictions=pred_fake_for_g,
            )
            loss_g_total.backward()
            opt_g.step()

            epoch_g_total += float(loss_g_total.item())
            epoch_g_adv += float(loss_g_adv.item())
            epoch_g_l1 += float(loss_g_l1.item())
            epoch_d_total += float(loss_d.item())
            num_batches += 1

            if batch_idx % log_interval == 0:
                print(
                    f"[Epoch {epoch}/{epochs}] [Batch {batch_idx}/{len(train_loader)}] "
                    f"D_loss: {loss_d.item():.4f} "
                    f"G_loss: {loss_g_total.item():.4f} "
                    f"(G_adv: {loss_g_adv.item():.4f}, G_L1: {loss_g_l1.item():.4f})"
                )

            # Save training samples
            if epoch % sample_interval == 0 and batch_idx == 0:
                save_comparison_grid(
                    printed=printed,
                    real_handwritten=real_hw,
                    fake_handwritten=fake_hw,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    save_dir=samples_dir / "train",
                )

        # End of epoch
        avg_g_total = epoch_g_total / max(1, num_batches)
        avg_g_adv = epoch_g_adv / max(1, num_batches)
        avg_g_l1 = epoch_g_l1 / max(1, num_batches)
        avg_d_total = epoch_d_total / max(1, num_batches)
        history["G_total"].append(avg_g_total)
        history["G_adv"].append(avg_g_adv)
        history["G_L1"].append(avg_g_l1)
        history["D_total"].append(avg_d_total)

        # Validation
        val_l1 = validate(
            generator=gen,
            val_loader=val_loader,
            criterion_l1=l1_val_loss,
            device=device,
            samples_dir=samples_dir,
            epoch=epoch,
        )
        history["val_L1"].append(val_l1)

        print(
            f"Epoch {epoch}/{epochs} completed. "
            f"avg_G_loss={avg_g_total:.4f}, avg_G_adv={avg_g_adv:.4f}, "
            f"avg_G_L1={avg_g_l1:.4f}, avg_D_loss={avg_d_total:.4f}, val_L1={val_l1:.4f}"
        )

        # Checkpointing
        if epoch % checkpoint_interval == 0 or epoch == epochs:
            ckpt_path = checkpoint_dir / f"pix2pix_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "generator_state_dict": gen.state_dict(),
                    "discriminator_state_dict": disc.state_dict(),
                    "optimizer_g_state_dict": opt_g.state_dict(),
                    "optimizer_d_state_dict": opt_d.state_dict(),
                    "history": history,
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")

        # Save training curves data + plot so far
        with curves_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        plot_training_curves(history, curves_plot_path)


if __name__ == "__main__":
    train()