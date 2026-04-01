from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from backend.latex.renderer import LatexRenderer


class PairedEquationDataset(Dataset):
    """
    Dataset of paired printed and handwritten equation images.
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

        # Ensure directory existence is checked but not necessarily created here
        if not self.printed_dir.exists() or not self.handwritten_dir.exists():
             raise RuntimeError(f"Directories {self.printed_dir} or {self.handwritten_dir} do not exist.")

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

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
    """Generate diverse LaTeX equation strings for better model generalization."""
    templates = [
        lambda: rf"\frac{{{random.randint(1,20)}}}{{{random.randint(1,20)}}}",
        lambda: rf"E = mc^{{{random.randint(2,3)}}}",
        lambda: rf"\int_{{{random.randint(0,1)}}}^{{{random.randint(2,5)}}} x^{{{random.randint(2,4)}}} \, dx",
        lambda: rf"\sum_{{i={random.randint(0,1)}}}^{{n}} i^{{{random.randint(1,2)}}}",
        lambda: r"a^2 + b^2 = c^2",
        lambda: rf"\sqrt{{{random.randint(10,99)}}}",
        lambda: r"\lim_{x \to \infty} \frac{1}{x} = 0",
        lambda: rf"f(x) = {random.randint(1,9)}x^2 + {random.randint(1,9)}x + {random.randint(1,9)}"
    ]
    return random.choice(templates)()


def _elastic_deform(image: np.ndarray, alpha: float = 36.0, sigma: float = 6.0) -> np.ndarray:
    """Apply elastic deformation to simulate natural handwriting jitters."""
    # Ensure 2D for processing
    if image.ndim == 3:
        img = np.squeeze(image)
    else:
        img = image

    h, w = img.shape[:2]
    
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def _add_noise_and_texture(image: np.ndarray) -> np.ndarray:
    """Add random noise and simulate pen pressure variations."""
    # Defensive check: Ensure image is 2D and uint8
    if image.ndim == 3:
        img = np.squeeze(image).copy()
    else:
        img = image.copy()
        
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    h, w = img.shape[:2] # This will no longer fail

    # 1. Random noise
    noise = np.random.normal(loc=0.0, scale=8.0, size=(h, w)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 2. Thickness variation (Dilate/Erode)
    ksize = random.choice([1, 2, 3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    if random.random() < 0.5:
        img = cv2.dilate(img, kernel, iterations=1)
    else:
        img = cv2.erode(img, kernel, iterations=1)

    # 3. Simulated stray strokes
    if random.random() < 0.3:
        for _ in range(random.randint(1, 4)):
            x1, y1 = random.randint(0, w-1), random.randint(0, h-1)
            x2, y2 = random.randint(0, w-1), random.randint(0, h-1)
            cv2.line(img, (x1, y1), (x2, y2), random.randint(200, 255), 1)

    return img


def create_synthetic_dataset(num_samples: int = 1000) -> Tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "data"
    printed_dir = data_root / "printed"
    handwritten_dir = data_root / "handwritten"

    # Clean old failed attempts to avoid name mismatches
    import shutil
    if printed_dir.exists():
        shutil.rmtree(printed_dir)
    if handwritten_dir.exists():
        shutil.rmtree(handwritten_dir)

    printed_dir.mkdir(parents=True, exist_ok=True)
    handwritten_dir.mkdir(parents=True, exist_ok=True)

    # Initialize renderer without an internal output_dir to stop it from
    # saving extra "latex_0000" files in /outputs/
    renderer = LatexRenderer()

    print(f"🎨 Generating {num_samples} synthetic pairs...")

    for idx in range(num_samples):
        eq = _random_equation()
        # WE DEFINE THE NAME ONCE HERE
        base_name = f"sample_{idx:05d}.png"

        try:
            # Get the numpy array (the [0,1] float version)
            img_arr = renderer.render_latex_to_image(eq)

            # Printed version (Clean)
            printed_img = (img_arr * 255.0).astype(np.uint8)

            # Handwritten version (Messy)
            angle = random.uniform(-3.0, 3.0)
            h, w = printed_img.shape
            rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated = cv2.warpAffine(
                printed_img,
                rot_mat,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )

            deformed = _elastic_deform(rotated)
            handwritten_img = _add_noise_and_texture(deformed)

            # SAVE BOTH WITH THE EXACT SAME FILENAME
            cv2.imwrite(str(printed_dir / base_name), printed_img)
            cv2.imwrite(str(handwritten_dir / base_name), handwritten_img)

            if idx % 100 == 0:
                print(f"   Stored pair: {base_name}")

        except Exception:
            continue

    return printed_dir, handwritten_dir


def get_dataloaders(
    batch_size: int = 16,
    train_split: float = 0.8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
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