from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from matplotlib import mathtext
from PIL import Image


class LatexRenderingError(Exception):
    """Raised when LaTeX cannot be parsed or rendered."""


@dataclass
class LatexRenderer:
    """
    Render LaTeX strings to normalized grayscale images suitable for ML pipelines.

    The renderer uses ``matplotlib.mathtext`` to rasterize LaTeX into a bitmap,
    converts it to a grayscale NumPy array with values in ``[0.0, 1.0]``, and
    resizes/pads the result to a fixed output size while preserving aspect ratio.
    """

    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "outputs" / "latex_renders"
    )
    image_size: Tuple[int, int] = (256, 256)
    dpi: int = 200
    font_size: int = 20

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Use bitmap mathtext parser which does not require a GUI backend.
        self._parser = mathtext.MathTextParser("Bitmap")
        self._counter: int = 0

    def render_latex_to_image(
        self,
        latex_string: str,
        output_size: Tuple[int, int] | None = None,
    ) -> np.ndarray:
        """
        Render a single LaTeX expression to a normalized grayscale image.

        The image is:
        - returned as a NumPy array of shape ``(H, W)`` with values in ``[0.0, 1.0]``,
        - saved as a PNG file in ``self.output_dir``.

        Parameters
        ----------
        latex_string:
            LaTeX expression to render (without surrounding ``$``).
        output_size:
            Desired output size as ``(height, width)``. If ``None``, defaults to
            ``self.image_size``.

        Returns
        -------
        np.ndarray
            Grayscale image array with values in ``[0.0, 1.0]``.

        Raises
        ------
        LatexRenderingError
            If the LaTeX expression cannot be parsed or rendered.
        """
        if not latex_string or not latex_string.strip():
            raise LatexRenderingError("Empty LaTeX string cannot be rendered.")

        target_h, target_w = output_size or self.image_size

        try:
            rgba, _ = self._parser.to_rgba(
                latex_string,
                dpi=self.dpi,
                fontsize=self.font_size,
            )
        except Exception as exc:  # mathtext can raise ValueError and others
            raise LatexRenderingError(f"Failed to render LaTeX: {latex_string!r}") from exc

        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise LatexRenderingError("Unexpected RGBA array shape from mathtext renderer.")

        # Composite onto a white background using alpha channel.
        rgba = np.asarray(rgba, dtype=np.float32)
        rgb = rgba[..., :3]
        alpha = rgba[..., 3:4] / 255.0
        white = np.ones_like(rgb) * 255.0
        composited = rgb * alpha + white * (1.0 - alpha)

        # Convert to grayscale [0,1].
        grayscale = np.dot(composited[..., :3], [0.299, 0.587, 0.114]) / 255.0
        grayscale = np.clip(grayscale, 0.0, 1.0)

        # Resize with aspect-ratio preservation and padding.
        resized = self._resize_and_pad(grayscale, target_h, target_w)

        # Save PNG file.
        file_path = self._next_output_path()
        image_to_save = Image.fromarray((resized * 255.0).astype(np.uint8), mode="L")
        file_path.parent.mkdir(parents=True, exist_ok=True)
        image_to_save.save(file_path)

        return resized

    def preprocess_image(self, image_array: np.ndarray) -> torch.Tensor:
        """
        Convert a grayscale image array to a normalized PyTorch tensor.

        Steps:
        - Ensures float32 and values in ``[0.0, 1.0]``.
        - Adds a channel dimension (``1, H, W``) if necessary.
        - Applies normalization with mean=0.5 and std=0.5.

        Parameters
        ----------
        image_array:
            Input image as a NumPy array of shape ``(H, W)`` or ``(H, W, 1)``.

        Returns
        -------
        torch.Tensor
            Normalized tensor of shape ``(1, H, W)`` suitable for model input.
        """
        if image_array.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D image array, got shape {image_array.shape}.")

        arr = image_array.astype(np.float32)
        # If not already normalized to [0,1], try to scale down using max value heuristics.
        if arr.max() > 1.0:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)

        if arr.ndim == 3:
            # Assume HWC, keep single channel if present.
            if arr.shape[2] == 1:
                arr = arr[..., 0]
            else:
                raise ValueError("Expected single-channel image for preprocessing.")

        tensor = torch.from_numpy(arr)  # (H, W)
        tensor = tensor.unsqueeze(0)  # (1, H, W)
        tensor = tensor.float()

        # Normalize: (x - mean) / std
        mean = 0.5
        std = 0.5
        tensor = (tensor - mean) / std
        return tensor

    def _resize_and_pad(self, img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resize while preserving aspect ratio and pad to target size."""
        h, w = img.shape[:2]
        if h == 0 or w == 0:
            raise LatexRenderingError("Rendered image has zero size.")

        scale = min(target_h / h, target_w / w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        pil_img = Image.fromarray((img * 255.0).astype(np.uint8), mode="L")
        pil_resized = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
        resized = np.asarray(pil_resized, dtype=np.float32) / 255.0

        # Pad with white background.
        canvas = np.ones((target_h, target_w), dtype=np.float32)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        canvas[top : top + new_h, left : left + new_w] = resized
        return canvas

    def _next_output_path(self) -> Path:
        """Generate a new output file path for a rendered LaTeX expression."""
        self._counter += 1
        filename = f"latex_{self._counter:04d}.png"
        return self.output_dir / filename


def batch_render_latex_to_images(
    latex_strings: Sequence[str],
    renderer: Optional[LatexRenderer] = None,
) -> List[np.ndarray]:
    """
    Batch render multiple LaTeX expressions to images.

    Parameters
    ----------
    latex_strings:
        Iterable of LaTeX strings to render.
    renderer:
        Optional existing ``LatexRenderer`` instance. If ``None``, a new
        instance with default settings is created.

    Returns
    -------
    list[np.ndarray]
        List of rendered grayscale images as NumPy arrays.

    Notes
    -----
    - Invalid LaTeX expressions are skipped, and their errors are printed.
    - The order of successful renders follows the input order, minus failures.
    """
    if renderer is None:
        renderer = LatexRenderer()

    results: List[np.ndarray] = []
    for idx, expr in enumerate(latex_strings):
        try:
            img = renderer.render_latex_to_image(expr)
        except LatexRenderingError as exc:
            # For a production system, consider integrating with structured logging instead.
            print(f"[batch_render] Skipping expression {idx} due to error: {exc}")
            continue
        results.append(img)
    return results

