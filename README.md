# RL Fruit Game — Suika with Reinforcement Learning

A custom [Gymnasium](https://gymnasium.farama.org/) environment for the Suika (Watermelon) Game, with value-based (DQN) and policy-based (PPO/REINFORCE) RL agents.

## What it Does

Simulates the Suika puzzle game: drop fruits into a container, same-type fruits merge into the next tier, and the goal is to maximise score before the stack overflows. Agents learn solely from merge rewards — no hand-crafted heuristics.

## Quick Start

```bash
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Play it yourself
python scripts/play_human.py   # arrow keys to aim, SPACE to drop
```

## Environment: `suika_env/SuikaEnv-v0`

```python
import suika_env
import gymnasium as gym

env = gym.make("suika_env/SuikaEnv-v0")
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)  # action ∈ [0, 31]
```

**Turn-based**: agents act only after the physics has fully settled.

| | Detail |
|---|---|
| Action space | `Discrete(32)` — drop x-column |
| Observation | `Dict` with padded fruit positions/types, current & next fruit |
| Reward | Merge score gained this step (canonical Suika triangular: 1, 3, 6, … 66) |
| Termination | Any fruit resting above the danger line |

## Fruit Chain

| # | Fruit | Radius | Score |
|---|-------|-------:|------:|
| 0 | Cherry | 12 | 1 |
| 1 | Strawberry | 18 | 3 |
| 2 | Grape | 24 | 6 |
| 3 | Dekopon | 30 | 10 |
| 4 | Persimmon | 38 | 15 |
| 5 | Apple | 46 | 21 |
| 6 | Pear | 56 | 28 |
| 7 | Peach | 66 | 36 |
| 8 | Pineapple | 78 | 45 |
| 9 | Melon | 90 | 55 |
| 10 | Watermelon | 104 | 66 *(pair vanishes)* |

## Evaluation

- **Throughput:** ~64 steps/sec on CPU (adequate for training)
- **Tests:** 37/37 passing (`pytest tests/ -v`)
- **SB3 `check_env`:** passes (no errors)

## Video Links

*(to be added)*

## Project Structure

```
suika_env/      Gymnasium env (env.py, world.py, render.py, fruits.py, config.py)
tests/          Unit + API tests
scripts/        Human play demo
```
