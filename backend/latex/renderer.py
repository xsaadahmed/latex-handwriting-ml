from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
# Force Matplotlib to use a headless backend for server/notebook environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import mathtext
from PIL import Image, ImageChops


class LatexRenderingError(Exception):
    """Raised when LaTeX cannot be parsed or rendered."""


@dataclass
class LatexRenderer:
    """
    Renders LaTeX strings to normalized grayscale images.
    Uses a robust crop-and-pad approach to ensure consistent output for ML models.
    """
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "printed"
    )
    image_size: Tuple[int, int] = (256, 256)
    dpi: int = 200
    font_size: int = 20

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._counter: int = 0

    def render_latex_to_image(self, latex_string: str, output_size: Tuple[int, int] | None = None) -> np.ndarray:
        """
        Renders a LaTeX string, crops to content, and pads to a fixed square size.
        """
        if not latex_string or not latex_string.strip():
            raise LatexRenderingError("Empty LaTeX string provided.")

        target_h, target_w = output_size or self.image_size

        try:
            # 1. Render to a large canvas to ensure we capture the whole expression
            fig = plt.figure(figsize=(6, 2), dpi=self.dpi)
            fig.text(0.5, 0.5, f"${latex_string}$", size=self.font_size, va="center", ha="center")

            # 2. Extract buffer
            fig.canvas.draw()
            rgba = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)  # Explicitly close to free memory

            # 3. Convert to Grayscale and perform a tight crop
            pil_img = Image.fromarray(rgba).convert("L")
            inverted_img = ImageChops.invert(pil_img)
            bbox = inverted_img.getbbox()

            if not bbox:
                raise LatexRenderingError(f"Render resulted in an empty image for: {latex_string}")

            cropped = pil_img.crop(bbox)
            
            # 4. Resize and pad while preserving aspect ratio
            normalized_arr = np.array(cropped) / 255.0
            resized = self._resize_and_pad(normalized_arr, target_h, target_w)

            # 5. Persistent storage
            file_path = self.output_dir / f"latex_{self._counter:05d}.png"
            self._counter += 1
            Image.fromarray((resized * 255.0).astype(np.uint8)).save(file_path)

            return resized

        except Exception as exc:
            if isinstance(exc, LatexRenderingError):
                raise
            raise LatexRenderingError(f"Failed to render: {latex_string}") from exc

    def preprocess_for_model(self, image_array: np.ndarray) -> torch.Tensor:
        """
        Converts a [0, 1] numpy array to a [-1, 1] PyTorch tensor of shape (1, H, W).
        """
        tensor = torch.from_numpy(image_array).float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)  # Add channel dimension: (1, H, W)
        
        # Normalize from [0, 1] to [-1, 1]
        return (tensor - 0.5) / 0.5

    def _resize_and_pad(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resizes preserving aspect ratio and pads with white (1.0)."""
        if img.ndim == 3:
            img = img.mean(axis=2)

        h, w = img.shape
        # Use a 0.85 multiplier to ensure a clean margin around the expression
        scale = min(target_h / h, target_w / w) * 0.85
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

        pil_img = Image.fromarray((img * 255.0).astype(np.uint8), mode="L")
        pil_resized = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
        resized_arr = np.asarray(pil_resized) / 255.0

        # Create white background canvas
        canvas = np.ones((target_h, target_w), dtype=np.float32)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized_arr
        return canvas


def batch_render_latex_to_images(
    latex_strings: Sequence[str],
    renderer: Optional[LatexRenderer] = None,
) -> List[np.ndarray]:
    """Efficiently renders a sequence of LaTeX strings."""
    renderer = renderer or LatexRenderer()
    outputs: List[np.ndarray] = []
    for idx, expr in enumerate(latex_strings):
        try:
            img = renderer.render_latex_to_image(expr)
            outputs.append(img)
        except LatexRenderingError as exc:
            print(f"[batch_render] Skipping index {idx}: {exc}")
    return outputs