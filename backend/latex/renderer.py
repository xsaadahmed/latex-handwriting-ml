from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib
# Force Matplotlib to use the 'Agg' backend for headless environments (like Colab)
matplotlib.use('Agg')
from matplotlib import mathtext
from PIL import Image

class LatexRenderingError(Exception):
    """Raised when LaTeX cannot be parsed or rendered."""

@dataclass
class LatexRenderer:
    """
    Render LaTeX strings to normalized grayscale images suitable for ML pipelines.
    """

    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "outputs" / "latex_renders"
    )
    image_size: Tuple[int, int] = (256, 256)
    dpi: int = 200
    font_size: int = 20

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # FIX: Changed "Bitmap" to "agg" to support Linux/Colab environments
        self._parser = mathtext.MathTextParser("agg")
        self._counter: int = 0

    def render_latex_to_image(
        self,
        latex_string: str,
        output_size: Tuple[int, int] | None = None,
    ) -> np.ndarray:
        if not latex_string or not latex_string.strip():
            raise LatexRenderingError("Empty LaTeX string cannot be rendered.")

        target_h, target_w = output_size or self.image_size

        try:
            # Wrap in $$ to ensure math mode is active
            math_expr = f"${latex_string}$"
            rgba, _ = self._parser.to_rgba(
                math_expr,
                dpi=self.dpi,
                fontsize=self.font_size,
            )
        except Exception as exc:
            raise LatexRenderingError(f"Failed to render LaTeX: {latex_string!r}") from exc

        # Composite onto a white background
        rgba = np.asarray(rgba, dtype=np.float32)
        rgb = rgba[..., :3]
        # Agg returns values 0-1 or 0-255 depending on version; normalize to 0-1
        if rgba.max() > 1.1:
            rgba = rgba / 255.0
            rgb = rgb / 255.0
            
        alpha = rgba[..., 3:4]
        white = np.ones_like(rgb)
        composited = rgb * alpha + white * (1.0 - alpha)

        # Convert to grayscale [0,1]
        grayscale = np.dot(composited[..., :3], [0.299, 0.587, 0.114])
        grayscale = np.clip(grayscale, 0.0, 1.0)

        # Resize and pad
        resized = self._resize_and_pad(grayscale, target_h, target_w)

        # Save PNG file
        file_path = self._next_output_path()
        image_to_save = Image.fromarray((resized * 255.0).astype(np.uint8), mode="L")
        image_to_save.save(file_path)

        return resized

    def _resize_and_pad(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = img.shape[:2]
        scale = min(target_h / h, target_w / w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

        pil_img = Image.fromarray((img * 255.0).astype(np.uint8), mode="L")
        pil_resized = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
        resized = np.asarray(pil_resized, dtype=np.float32) / 255.0

        canvas = np.ones((target_h, target_w), dtype=np.float32)
        top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas

    def _next_output_path(self) -> Path:
        self._counter += 1
        return self.output_dir / f"latex_{self._counter:04d}.png"

    def preprocess_image(self, image_array: np.ndarray) -> torch.Tensor:
        arr = image_array.astype(np.float32)
        if arr.max() > 1.0: arr /= 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).float()
        return (tensor - 0.5) / 0.5