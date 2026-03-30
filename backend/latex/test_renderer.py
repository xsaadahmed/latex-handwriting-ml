from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .renderer import LatexRenderer, batch_render_latex_to_images


def _get_test_output_dir() -> Path:
    """Return the path to the `outputs/latex_tests` directory (project-root relative)."""
    project_root = Path(__file__).resolve().parents[2]
    out_dir = project_root / "outputs" / "latex_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def run_sample_tests() -> None:
    """Render a set of sample LaTeX equations and save them to disk."""
    output_dir = _get_test_output_dir()

    expressions: List[str] = [
        r"\\int_0^1 x^2 \\, dx",
        r"\\frac{d}{dx} x^2",
        r"\\sum_{i=1}^n i^2",
        r"E = mc^2",
        r"\\nabla \\cdot \\vec{E} = \\frac{\\rho}{\\epsilon_0}",
    ]

    renderer = LatexRenderer(output_dir=output_dir)
    print(f"Using output directory: {output_dir}")

    images: List[np.ndarray] = batch_render_latex_to_images(expressions, renderer=renderer)

    for idx, (expr, img) in enumerate(zip(expressions, images), start=1):
        print(f"[Test {idx}] Rendered expression: {expr!r} -> shape={img.shape}, dtype={img.dtype}")

    print(f"Successfully rendered {len(images)} / {len(expressions)} expressions.")
    print(f"Check PNG files under: {output_dir}")


if __name__ == "__main__":
    run_sample_tests()

