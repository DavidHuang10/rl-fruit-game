# Setup

## Local

Use Python 3.10+.

```bash
uv sync
uv run pytest tests/ -v
```

If using `pip` instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

## Play

```bash
uv run play
```

Arrow keys move the drop column. Space drops the fruit.

## Local Training

```bash
uv run python scripts/train_dqn.py
uv run python scripts/train_ppo_sb3.py
```

Smoke tests:

```bash
uv run python scripts/train_dqn.py --total_steps 5000 --warmup 1000
uv run python scripts/train_ppo_sb3.py --total_timesteps 5000 --n_envs 1
```

## Cluster Training

From the project root on the cluster:

```bash
bash scripts/setup_cluster.sh
sbatch scripts/train_dqn_slurm.sh
sbatch scripts/train_ppo_slurm.sh
```

`uv sync` creates the project environment at `.venv`. The Slurm scripts run Python through `uv run`, so they use that same environment.

Logs/checkpoints are written under `results/dqn/` and `results/ppo/`.

## Evaluation

```bash
uv run python scripts/eval_agent.py --agent random --episodes 50
uv run python scripts/eval_agent.py --agent center --episodes 50
uv run python scripts/eval_agent.py --agent dqn --checkpoint models/dqn/model.pt --episodes 50
uv run python scripts/eval_agent.py --agent ppo --checkpoint models/ppo/model.zip --episodes 50
```

Metrics saved:

- episode return
- final game score
- episode length
- final fruit count
- final physics-frame count

Outputs go to `results/eval/`. Legacy checkpoint paths under `results/dqn/` and `results/ppo/` are compatibility links to `models/`.
