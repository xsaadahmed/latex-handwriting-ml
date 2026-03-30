from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from backend.latex import LatexRenderer


class PairedEquationDataset(Dataset):
    """
    Dataset of paired printed and handwritten equation images.

    Assumes that both directories contain images with matching filenames,
    e.g. `eq_00001.png` exists in both `printed_dir` and `handwritten_dir`.
    """

    def __init__(
        self,
        printed_dir: str | Path,
        handwritten_dir: str | Path,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        super().__init__()
        self.printed_dir = Path(printed_dir)
        self.handwritten_dir = Path(handwritten_dir)

        self.printed_dir.mkdir(parents=True, exist_ok=True)
        self.handwritten_dir.mkdir(parents=True, exist_ok=True)

        printed_files = {p.name for p in self.printed_dir.glob("*.png")}
        handwritten_files = {p.name for p in self.handwritten_dir.glob("*.png")}
        self.filenames: List[str] = sorted(printed_files & handwritten_files)

        if not self.filenames:
            raise RuntimeError(
                f"No paired PNG files found in {self.printed_dir} and {self.handwritten_dir}."
            )

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),  # [0,1]
                    transforms.Normalize(mean=[0.5], std=[0.5]),  # -> [-1,1]
                ]
            )
        else:
            self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        fname = self.filenames[idx]
        printed_path = self.printed_dir / fname
        handwritten_path = self.handwritten_dir / fname

        printed_img = Image.open(printed_path).convert("L")
        handwritten_img = Image.open(handwritten_path).convert("L")

        printed_tensor = self.transform(printed_img)
        handwritten_tensor = self.transform(handwritten_img)

        return {
            "printed": printed_tensor,
            "handwritten": handwritten_tensor,
            "filename": fname,
        }


def _random_equation() -> str:
    """Generate a simple random LaTeX equation string."""
    ops = [r"+", r"-", r"\times", r"\div"]
    idx = random.randint(0, 3)
    if idx == 0:
        a, b = random.randint(1, 9), random.randint(1, 9)
        return rf"{a}x^{b} + {b}x + {a}"
    if idx == 1:
        n = random.randint(3, 10)
        return rf"\sum_{{i=1}}^{{{n}}} i^2"
    if idx == 2:
        a, b = random.randint(1, 5), random.randint(1, 5)
        return rf"\int_0^1 x^{a}\, dx = \frac1{{{a+1}}}"
    op = random.choice(ops)
    a, b = random.randint(1, 9), random.randint(1, 9)
    return rf"{a} {op} {b} = ?"


def _elastic_deform(image: np.ndarray, alpha: float = 36.0, sigma: float = 6.0) -> np.ndarray:
    """
    Apply a simple elastic deformation using OpenCV.

    This is a lightweight approximation suitable for handwriting-style warps.
    """
    if image.ndim == 3:
        h, w, c = image.shape
    else:
        h, w = image.shape
        c = 1
        image = image[..., None]

    dx = cv2.GaussianBlur((np.random.rand(h, w, 1) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w, 1) * 2 - 1), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx[..., 0]).astype(np.float32)
    map_y = (y + dy[..., 0]).astype(np.float32)

    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if c == 1:
        warped = warped[..., 0]
    return warped


def _add_noise_and_texture(image: np.ndarray) -> np.ndarray:
    """Add random noise, line strokes, and thickness variations."""
    img = image.copy().astype(np.uint8)
    h, w = img.shape[:2]

    # Random noise
    noise = np.random.normal(loc=0.0, scale=10.0, size=(h, w)).astype(np.int16)
    noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Random thickness variations using dilation/erosion.
    ksize = random.choice([1, 2, 3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if random.random() < 0.5:
        noisy = cv2.dilate(noisy, kernel, iterations=1)
    else:
        noisy = cv2.erode(noisy, kernel, iterations=1)

    # Random strokes / scribbles.
    num_lines = random.randint(3, 10)
    for _ in range(num_lines):
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        x2, y2 = random.randint(0, w - 1), random.randint(0, h - 1)
        color = random.randint(150, 255)
        thickness = random.randint(1, 2)
        cv2.line(noisy, (x1, y1), (x2, y2), int(color), thickness=thickness)

    return noisy


def create_synthetic_dataset(num_samples: int = 1000) -> Tuple[Path, Path]:
    """
    Create a synthetic paired dataset using the LaTeX renderer and heuristic augmentations.

    Returns
    -------
    printed_dir, handwritten_dir
        Directories containing paired PNG images.
    """
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    printed_dir = data_root / "printed"
    handwritten_dir = data_root / "handwritten"
    printed_dir.mkdir(parents=True, exist_ok=True)
    handwritten_dir.mkdir(parents=True, exist_ok=True)

    renderer = LatexRenderer(output_dir=project_root / "outputs" / "synthetic_printed")

    for idx in range(num_samples):
        eq = _random_equation()
        base_name = f"eq_{idx:05d}.png"

        try:
            img_arr = renderer.render_latex_to_image(eq)
        except Exception:
            # Skip invalid renderings.
            continue

        # Convert to uint8 grayscale [0,255].
        base_img = (img_arr * 255.0).astype(np.uint8)

        # Printed version: slight blur only.
        printed = cv2.GaussianBlur(base_img, (3, 3), 0)

        # Handwritten-like version: rotation, elastic, noise/texture.
        angle = random.uniform(-5.0, 5.0)
        h, w = base_img.shape
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(base_img, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        deformed = _elastic_deform(rotated)
        handwritten = _add_noise_and_texture(deformed)

        cv2.imwrite(str(printed_dir / base_name), printed)
        cv2.imwrite(str(handwritten_dir / base_name), handwritten)

    return printed_dir, handwritten_dir


def get_dataloaders(
    batch_size: int = 16,
    train_split: float = 0.8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Construct training and validation DataLoaders for paired equation images.

    Assumes that `data/printed` and `data/handwritten` contain paired PNG files.
    """
    project_root = Path(__file__).resolve().parents[2]
    printed_dir = project_root / "data" / "printed"
    handwritten_dir = project_root / "data" / "handwritten"

    dataset = PairedEquationDataset(printed_dir=printed_dir, handwritten_dir=handwritten_dir)

    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = n_total - n_train
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader

