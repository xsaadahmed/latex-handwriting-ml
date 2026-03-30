from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def _to_display(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a normalized tensor (NCHW or CHW) in [-1,1] to [0,1] for display.
    """
    if tensor.dim() == 4:
        t = tensor
    elif tensor.dim() == 3:
        t = tensor.unsqueeze(0)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape={tuple(tensor.shape)}")
    t = (t * 0.5) + 0.5
    return t.clamp(0.0, 1.0)


def plot_training_curves(losses_dict: Dict[str, List[float]], save_path: str | Path) -> None:
    """
    Plot generator and discriminator losses (and optional metrics) over epochs.

    Parameters
    ----------
    losses_dict:
        Dictionary with keys like "G_total", "D_total", "val_L1", each mapping
        to a list of per-epoch scalar values.
    save_path:
        Output PNG path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for key, values in losses_dict.items():
        if not values:
            continue
        plt.plot(range(1, len(values) + 1), values, label=key)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def save_comparison_grid(
    printed: torch.Tensor,
    real_handwritten: torch.Tensor | None,
    fake_handwritten: torch.Tensor,
    epoch: int,
    batch_idx: int,
    save_dir: str | Path,
) -> None:
    """
    Save a side-by-side comparison of printed, real handwritten, and generated images.

    Expects batched tensors of shape (B, 1, H, W) in normalized [-1,1] range.
    Only the first example in the batch is visualized.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    printed_1 = printed[0:1]
    fake_1 = fake_handwritten[0:1]
    images: List[torch.Tensor] = [printed_1]

    if real_handwritten is not None:
        real_1 = real_handwritten[0:1]
        images.append(real_1)
    images.append(fake_1)

    grid = make_grid(_to_display(torch.cat(images, dim=0)), nrow=len(images), padding=5)

    plt.figure(figsize=(4 * len(images), 4))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    fname = f"epoch_{epoch:03d}_batch_{batch_idx:04d}.png"
    plt.tight_layout()
    plt.savefig(str(save_dir / fname), bbox_inches="tight", pad_inches=0.1)
    plt.close()


def visualize_results(
    model: torch.nn.Module,
    test_samples: torch.Tensor,
    save_path: str | Path,
) -> None:
    """
    Run the model on test samples and save a grid of (input, output) pairs.

    Parameters
    ----------
    model:
        Trained generator model.
    test_samples:
        Tensor of printed equations with shape (B, 1, H, W) in normalized [-1,1].
    save_path:
        Path where the PNG grid will be written.
    """
    device = next(model.parameters()).device
    test_samples = test_samples.to(device)

    model.eval()
    with torch.no_grad():
        generated = model(test_samples)

    # Interleave printed and generated: [p1,g1,p2,g2,...]
    pairs: List[torch.Tensor] = []
    for p, g in zip(test_samples, generated):
        pairs.append(p.unsqueeze(0))
        pairs.append(g.unsqueeze(0))
    stacked = torch.cat(pairs, dim=0)

    grid = make_grid(_to_display(stacked), nrow=2, padding=5)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4 * (len(test_samples) // 2 + 1)))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.tight_layout()
    plt.savefig(str(save_path), bbox_inches="tight", pad_inches=0.1)
    plt.close()

