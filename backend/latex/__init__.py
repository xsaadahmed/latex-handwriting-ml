"""
LaTeX utilities for the LaTeX Handwriting ML project.

Exposes the high-level `LatexRenderer` class and batch rendering helper.
"""

from .renderer import LatexRenderer, LatexRenderingError, batch_render_latex_to_images

__all__ = ["LatexRenderer", "LatexRenderingError", "batch_render_latex_to_images"]

