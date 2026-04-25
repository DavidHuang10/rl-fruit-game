#!/usr/bin/env bash
#SBATCH --job-name=suika-dqn
#SBATCH --output=results/dqn/slurm_%j.out
#SBATCH --error=results/dqn/slurm_%j.err
#SBATCH --partition=scavenger-gpu        # free tier; change to gpu-common for priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00                  # 500k steps takes ~1-2h on GPU; 6h is safe

# ── environment ───────────────────────────────────────────────────────────────
module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || \
  module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
  module load python/3.11 2>/dev/null || true

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/.venv_cluster/bin/activate"

# Headless SDL — prevents pygame from trying to open a display window
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

# ── diagnostics ───────────────────────────────────────────────────────────────
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Time:     $(date)"
echo "Dir:      $PROJECT_ROOT"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

mkdir -p "$PROJECT_ROOT/results/dqn"

# ── training ──────────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"
python scripts/train_dqn.py \
  --total_steps 500000 \
  --warmup      5000   \
  --buffer_cap  100000 \
  --batch_size  128    \
  --grad_freq   4      \
  --log_freq    5000   \
  --ckpt_freq   50000  \
  --seed        42

echo "Done: $(date)"
