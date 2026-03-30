from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, request
from flask_cors import CORS

from backend.latex import LatexRenderer


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    This is a minimal scaffold that exposes a health check and a LaTeX
    rendering endpoint. It can later be extended to run full model inference.
    """
    app = Flask(__name__)
    CORS(app)

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "outputs" / "api_renders"
    renderer = LatexRenderer(output_dir=output_dir)

    @app.get("/health")
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.post("/render-latex")
    def render_latex() -> Any:
        payload: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        latex = payload.get("latex")
        if not isinstance(latex, str) or not latex.strip():
            return jsonify({"error": "Field 'latex' must be a non-empty string."}), 400

        try:
            img = renderer.render_latex_to_image(latex)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400

        # For now we only return basic metadata; the PNG is saved on disk.
        return jsonify(
            {
                "latex": latex,
                "shape": list(img.shape),
                "output_dir": str(output_dir),
            }
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)

