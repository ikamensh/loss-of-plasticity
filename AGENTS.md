# Agent Instructions

## Focus area
I'm building the script `drift/dynamic_mnist_cbp.py`, unless stated otherwise assume tasks
are about this script and its dependencies.
Put new files related to it nearby in `drift` folder.

## Environment Setup
- Python 3.12 with CPU-only PyTorch 2.8.0 and torchvision 0.23.0 are installed via `pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu`.
- Additional dependencies: `tqdm`, `PyYAML`, `scipy`, `mlproj-manager`, `zipp`, `pycparser`, `gym`, and friends were installed with `pip`.
- The repository can be used without installation, but running `pip install -e .` exposes the `lop` package globally.

## Working with Scripts
- Scripts run modules from the `lop` package. When adding new scripts, append the repository root to `sys.path` so they work without installing the package.
- Example training script: `python drift/dynamic_mnist_cbp.py` trains a tiny conv+linear network on a MNIST stream where class proportions perform a random walk. Pass `--cbp` to enable Continuous Backprop layers or omit it for a vanilla baseline.
- The MNIST dataset is downloaded automatically under `data/`, which is git-ignored.

## Checks
- After modifying training logic or CBP layers, run `python drift/dynamic_mnist_cbp.py --epochs 1` (and optionally `--cbp`) to ensure the model still trains and reports reasonable accuracy.
