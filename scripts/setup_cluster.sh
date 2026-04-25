#!/usr/bin/env bash
# Run once on the DCC login node to create a venv and install dependencies.
# Usage:  bash scripts/setup_cluster.sh
#
# After this completes, your venv lives at $PROJECT_ROOT/.venv_cluster
# and the SLURM script activates it automatically.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$PROJECT_ROOT/.venv_cluster"

echo "=== Setting up Python environment at $VENV ==="

# Load modules available on Duke DCC
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || \
  module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
  module load python/3.11 2>/dev/null || \
  echo "WARNING: Could not load a Python module — using system Python"

python3 --version

python3 -m venv "$VENV"
source "$VENV/bin/activate"

pip install --upgrade pip wheel

# PyTorch with CUDA 12.1 (matches DCC A100/V100 nodes)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Project deps (no uv needed on cluster)
pip install \
  pymunk>=6.6 \
  "pygame>=2.5" \
  "gymnasium>=0.29" \
  "numpy>=1.26" \
  "stable-baselines3>=2.0" \
  "matplotlib>=3.7" \
  "pandas>=2.0"

# Install the project package itself so suika_env / agents imports work
pip install -e "$PROJECT_ROOT"

echo ""
echo "=== Setup complete ==="
echo "Venv: $VENV"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
