#!/usr/bin/env bash
#SBATCH --job-name=suika-ppo-v2
#SBATCH --output=results/ppo_v2/slurm_%j.out
#SBATCH --error=results/ppo_v2/slurm_%j.err
#SBATCH --partition=scavenger-gpu        # free tier; change to gpu-common for priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00

module load Python/3.11.5-GCCcore-13.2.0 2>/dev/null || \
  module load Python/3.11.3-GCCcore-12.3.0 2>/dev/null || \
  module load python/3.11 2>/dev/null || true

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"

export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export PYGAME_HIDE_SUPPORT_PROMPT=1

echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Time:     $(date)"
echo "Dir:      $PROJECT_ROOT"
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

mkdir -p "$PROJECT_ROOT/results/ppo_v2"

cd "$PROJECT_ROOT"
uv run python scripts/train_ppo_sb3.py \
  --out_dir         results/ppo_v2 \
  --total_timesteps 1000000 \
  --n_envs          4      \
  --n_steps         2048   \
  --batch_size      256    \
  --learning_rate   0.0001 \
  --gamma           0.99   \
  --gae_lambda      0.95   \
  --ent_coef        0.001  \
  --log_freq        5000   \
  --seed            42

echo "Done: $(date)"
