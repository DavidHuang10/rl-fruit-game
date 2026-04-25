# RL Fruit Game

Suika-style fruit merging implemented as a custom Gymnasium environment, with DQN, PPO, random, and center-drop agents for comparison.

## What It Does

The environment simulates a turn-based Suika puzzle game. The agent chooses one of 32 drop columns, physics resolves, matching fruits merge into larger fruits, and the episode ends when the stack crosses the danger line. Rewards are based on merge score only.

## Quick Start

```bash
uv sync
uv run pytest tests/ -v
uv run python scripts/play_human.py
```

Human controls: arrow keys move the drop column, space drops the fruit.

## Train

```bash
# DQN
uv run python scripts/train_dqn.py

# PPO
uv run python scripts/train_ppo_sb3.py
```

Cluster jobs:

```bash
sbatch scripts/train_dqn_slurm.sh
sbatch scripts/train_ppo_slurm.sh
```

Outputs are written to `results/dqn/` and `results/ppo/`.

## Evaluate

```bash
uv run python scripts/eval_agent.py --agent random --episodes 50
uv run python scripts/eval_agent.py --agent center --episodes 50
uv run python scripts/eval_agent.py --agent dqn --checkpoint results/dqn/model.pt --episodes 50
uv run python scripts/eval_agent.py --agent ppo --checkpoint results/ppo/model.zip --episodes 50
```

Evaluation writes per-episode CSVs and summary JSON files to `results/eval/`.

## Current Status

- DQN and PPO training are running on the cluster.
- Random and center baselines have 50-episode evaluation summaries.
- Current PPO logs do not show clear learning: returns are mostly flat, entropy remains near random-action levels, and value explained variance is near zero.
- Final DQN/PPO comparison should use the completed cluster checkpoints.

## Environment

```python
import gymnasium as gym
import suika_env

env = gym.make("suika_env/SuikaEnv-v0")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

| Property | Value |
|---|---|
| Action space | `Discrete(32)` drop column |
| Observation | Dict with padded fruit positions/types, fruit mask, current fruit, next fruit |
| Reward | Merge score gained after the drop |
| Termination | Fruit stack crosses the danger line |

## Project Structure

```text
suika_env/   Gymnasium environment and physics
agents/      DQN, replay buffer, network, baseline agents
scripts/     Training, evaluation, human play, Slurm jobs
tests/       Unit and API tests
results/     Checkpoints, metrics, plots, eval summaries
```
