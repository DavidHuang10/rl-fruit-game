#!/usr/bin/env bash
# Run once on the DCC login node to install dependencies via uv.
# Usage:  bash scripts/setup_cluster.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Install uv if not already present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== uv $(uv --version) ==="
uv sync

echo ""
echo "=== Setup complete ==="
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
