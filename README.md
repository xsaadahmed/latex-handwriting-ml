# LaTeX Handwriting ML

This project aims to generate **handwritten-style images** from LaTeX expressions using deep learning models (e.g. GAN-style architectures).

### Project layout

- `backend/`: Model components, LaTeX rendering utilities, API server, and helpers.
- `data/`: Printed and handwritten datasets, style references, and sample LaTeX.
- `checkpoints/`: Saved model weights and training snapshots.
- `outputs/`: Generated images, evaluation artifacts, and test renders.
- `frontend/`: (To be implemented) Web UI for interacting with the service.
- `config.yaml`: Central configuration for paths, image dimensions, and hyperparameters.

### Getting started

1. **Create a virtual environment** (recommended).
2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the LaTeX renderer test**:

```bash
python -m backend.latex.test_renderer
```

This will render a set of sample LaTeX expressions and save the outputs under `outputs/latex_tests/`.

### Notes

- The core deep learning models in `backend/model/` are scaffolds/placeholders and can be extended with your preferred architecture.
- The LaTeX renderer in `backend/latex/renderer.py` uses `matplotlib.mathtext` and is designed for production-style robustness (type hints, docstrings, error handling).
