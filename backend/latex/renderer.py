from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
# Force Matplotlib to use a headless backend.
matplotlib.use("Agg")
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
        self._parser = mathtext.MathTextParser("Agg")
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
            # Wrap in $$ if not present
            math_expr = f"${latex_string}$" if not latex_string.startswith("$") else latex_string

            # MODERN WAY: parse then render
            width, height, depth, glyphs, rects = self._parser.parse(
                math_expr, dpi=self.dpi, prop=matplotlib.font_manager.FontProperties(size=self.font_size)
            )

            # Create a figure to draw the parsed math
            fig = matplotlib.figure.Figure(figsize=(width / self.dpi, height / self.dpi), dpi=self.dpi)
            fig.patch.set_alpha(0)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()

            # Draw the math text
            ax.text(
                0,
                depth / height,
                math_expr,
                fontproperties=matplotlib.font_manager.FontProperties(size=self.font_size),
            )

            # Convert figure to RGBA array
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
            canvas.draw()
            rgba = np.asarray(canvas.buffer_rgba())
        except Exception as exc:
            raise LatexRenderingError(f"Failed to render LaTeX: {latex_string!r}") from exc

        # --- Remaining logic (Compositing, Grayscale, Resize) remains the same ---
        rgba = rgba.astype(np.float32) / 255.0
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4]
        white = np.ones_like(rgb)
        composited = rgb * alpha + white * (1.0 - alpha)

        grayscale = np.dot(composited[..., :3], [0.299, 0.587, 0.114])
        grayscale = np.clip(grayscale, 0.0, 1.0)

        resized = self._resize_and_pad(grayscale, target_h, target_w)

        file_path = self._next_output_path()
        image_to_save = Image.fromarray((resized * 255.0).astype(np.uint8), mode="L")
        image_to_save.save(file_path)

        return resized

    def _resize_and_pad(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize while preserving aspect ratio and pad to the target canvas."""
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise LatexRenderingError("Rendered image has invalid shape.")

        scale = min(target_h / h, target_w / w)
        new_h, new_w = max(1, int(round(h * scale))), max(1, int(round(w * scale)))

        pil_img = Image.fromarray((img * 255.0).astype(np.uint8), mode="L")
        pil_resized = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
        resized = np.asarray(pil_resized, dtype=np.float32) / 255.0

        canvas = np.ones((target_h, target_w), dtype=np.float32)
        top, left = (target_h - new_h) // 2, (target_w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas

    def _next_output_path(self) -> Path:
        """Return the next auto-incremented output path."""
        self._counter += 1
        return self.output_dir / f"latex_{self._counter:04d}.png"

    def preprocess_image(self, image_array: np.ndarray) -> torch.Tensor:
        """Convert grayscale image array to normalized tensor with shape (1, H, W)."""
        arr = image_array.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        arr = np.clip(arr, 0.0, 1.0)

        tensor = torch.from_numpy(arr).unsqueeze(0).float()
        return (tensor - 0.5) / 0.5


def batch_render_latex_to_images(
    latex_strings: Sequence[str],
    renderer: Optional[LatexRenderer] = None,
) -> List[np.ndarray]:
    """Render multiple LaTeX expressions; skip invalid ones with logged errors."""
    renderer = renderer or LatexRenderer()
    outputs: List[np.ndarray] = []
    for idx, expr in enumerate(latex_strings):
        try:
            outputs.append(renderer.render_latex_to_image(expr))
        except LatexRenderingError as exc:
            print(f"[batch_render] Skipping index {idx}: {exc}")
    return outputs