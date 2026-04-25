# Generated with Claude Code (claude-sonnet-4-6). Training loop, hyperparameters, and
# evaluation cadence directed by David Huang.

"""Train a Double DQN agent on SuikaEnv.

Usage:
    uv run python scripts/train_dqn.py
    uv run python scripts/train_dqn.py --total_steps 5000 --warmup 1000   # smoke test
    uv run python scripts/train_dqn.py --n_envs 4                          # parallel envs
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from suika_env import SuikaEnv
from agents.dqn import DQNAgent
from agents.replay_buffer import DictReplayBuffer


RESULTS_DIR = pathlib.Path("results/dqn")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total_steps",   type=int,   default=500_000)
    p.add_argument("--warmup",        type=int,   default=5_000)
    p.add_argument("--buffer_cap",    type=int,   default=100_000)
    p.add_argument("--batch_size",    type=int,   default=128)
    p.add_argument("--grad_freq",     type=int,   default=4,      help="gradient step every N env steps")
    p.add_argument("--log_freq",      type=int,   default=5_000,  help="log interval (env steps)")
    p.add_argument("--ckpt_freq",     type=int,   default=50_000, help="checkpoint interval (env steps)")
    p.add_argument("--n_envs",        type=int,   default=1,      help="parallel environments via AsyncVectorEnv")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def _crossed(env_steps: int, n_envs: int, freq: int) -> bool:
    """True when env_steps just crossed a multiple of freq."""
    return env_steps // freq > (env_steps - n_envs) // freq


def _save_curves(metrics_path: pathlib.Path) -> None:
    """Read metrics.csv and write learning_curve.png alongside it."""
    env_steps, ep_returns, losses = [], [], []
    with open(metrics_path) as f:
        for row in csv.DictReader(f):
            env_steps.append(int(row["env_step"]))
            ep_returns.append(float(row["mean_ep_return"]))
            if row["mean_loss"]:
                losses.append(float(row["mean_loss"]))
            else:
                losses.append(float("nan"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(env_steps, ep_returns, linewidth=1)
    ax1.set_ylabel("Mean episode return (last 100 eps)")
    ax1.set_title("DQN — Suika training curves")
    ax2.plot(env_steps, losses, linewidth=1, color="orange")
    ax2.set_ylabel("Mean Huber loss")
    ax2.set_xlabel("Env steps")
    fig.tight_layout()
    fig.savefig(metrics_path.parent / "learning_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    n_envs = args.n_envs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    if n_envs > 1:
        from gymnasium.vector import AsyncVectorEnv
        env = AsyncVectorEnv([lambda: SuikaEnv() for _ in range(n_envs)])
        obs, _ = env.reset(seed=list(range(args.seed, args.seed + n_envs)))
    else:
        env = SuikaEnv()
        obs, _ = env.reset(seed=args.seed)

    agent  = DQNAgent(
        batch_size=args.batch_size,
        total_grad_steps=max(1, (args.total_steps - args.warmup) // args.grad_freq),
    )
    buffer = DictReplayBuffer(capacity=args.buffer_cap)

    print(f"Device:      {agent.device}")
    print(f"Envs:        {n_envs}")
    print(f"Total steps: {args.total_steps:,}  |  warmup: {args.warmup:,}")
    print(f"Results →    {RESULTS_DIR.resolve()}\n")

    metrics_path = RESULTS_DIR / "metrics.csv"
    csv_file = open(metrics_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "env_step", "mean_ep_return", "mean_ep_score", "mean_ep_length",
        "mean_merges", "mean_loss", "epsilon", "lr",
    ])
    writer.writeheader()

    ep_returns: collections.deque[float] = collections.deque(maxlen=100)
    ep_scores:  collections.deque[float] = collections.deque(maxlen=100)
    ep_lengths: collections.deque[int]   = collections.deque(maxlen=100)
    recent_losses: list[float]           = []

    # scalar for single env, array for vectorized — keeps types clean
    if n_envs > 1:
        ep_ret: float | np.ndarray = np.zeros(n_envs)
        ep_len: int | np.ndarray   = np.zeros(n_envs, dtype=int)
    else:
        ep_ret = 0.0
        ep_len = 0
    t0 = time.time()

    # grad step every ceil(grad_freq / n_envs) loop iterations
    grad_iter_freq = max(1, args.grad_freq // n_envs)

    for iter_num in range(1, args.total_steps // n_envs + 1):
        env_steps = iter_num * n_envs
        in_warmup = env_steps <= args.warmup

        # ── action selection ──────────────────────────────────────────────
        if in_warmup:
            action = env.action_space.sample()
        elif n_envs > 1:
            action = np.array([
                agent.select_action({k: v[i] for k, v in obs.items()})
                for i in range(n_envs)
            ])
        else:
            action = agent.select_action(obs)

        # ── env step ──────────────────────────────────────────────────────
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated

        # ── buffer push ───────────────────────────────────────────────────
        if n_envs > 1:
            for i in range(n_envs):
                obs_i  = {k: v[i] for k, v in obs.items()}
                nobs_i = {k: v[i] for k, v in next_obs.items()}
                buffer.push(obs_i, int(action[i]), float(reward[i]), nobs_i, bool(done[i]))
        else:
            buffer.push(obs, int(action), float(reward), next_obs, bool(done))

        agent.tick_env_step(n_envs)

        # ── episode tracking ──────────────────────────────────────────────
        ep_ret += reward
        ep_len += 1

        if n_envs > 1:
            for i in range(n_envs):
                if done[i]:
                    ep_returns.append(float(ep_ret[i]))
                    ep_lengths.append(int(ep_len[i]))
                    ep_ret[i] = 0.0
                    ep_len[i] = 0
        else:
            if done:
                ep_returns.append(float(ep_ret))  # type: ignore[arg-type]
                ep_scores.append(float(info.get("score", 0.0)))
                ep_lengths.append(int(ep_len))    # type: ignore[arg-type]
                ep_ret = 0.0
                ep_len = 0
                next_obs, _ = env.reset()

        obs = next_obs

        # ── gradient step ─────────────────────────────────────────────────
        if not in_warmup and iter_num % grad_iter_freq == 0 and buffer.size >= args.batch_size:
            loss = agent.update(buffer)
            recent_losses.append(loss)

        # ── logging ───────────────────────────────────────────────────────
        if _crossed(env_steps, n_envs, args.log_freq):
            mean_ret   = float(np.mean(ep_returns))    if ep_returns   else float("nan")
            mean_score = float(np.mean(ep_scores))     if ep_scores    else float("nan")
            mean_len   = float(np.mean(ep_lengths))    if ep_lengths   else float("nan")
            mean_loss  = float(np.mean(recent_losses)) if recent_losses else float("nan")
            current_lr = agent.scheduler.get_last_lr()[0]
            recent_losses.clear()

            writer.writerow({
                "env_step":       env_steps,
                "mean_ep_return": f"{mean_ret:.2f}",
                "mean_ep_score":  f"{mean_score:.2f}",
                "mean_ep_length": f"{mean_len:.1f}",
                "mean_merges":    "",
                "mean_loss":      f"{mean_loss:.6f}" if not np.isnan(mean_loss) else "",
                "epsilon":        f"{agent.epsilon:.4f}",
                "lr":             f"{current_lr:.2e}",
            })
            csv_file.flush()

            elapsed = time.time() - t0
            steps_per_sec = env_steps / elapsed
            eta_sec = (args.total_steps - env_steps) / max(steps_per_sec, 1e-9)
            print(
                f"step {env_steps:>8,} | ret {mean_ret:>8.1f} | score {mean_score:>8.1f} | "
                f"loss {mean_loss:>9.5f} | eps {agent.epsilon:.3f} | "
                f"lr {current_lr:.2e} | {steps_per_sec:.1f} sps | ETA {eta_sec/60:.0f}m"
            )

        # ── checkpoint ────────────────────────────────────────────────────
        if _crossed(env_steps, n_envs, args.ckpt_freq):
            agent.save(RESULTS_DIR / f"model_step{env_steps}.pt")

    # ── final save ────────────────────────────────────────────────────────
    csv_file.close()
    env.close()
    agent.save(RESULTS_DIR / "model.pt")
    _save_curves(metrics_path)
    print(f"\nDone. Model → {RESULTS_DIR / 'model.pt'}")
    print(f"Curves → {RESULTS_DIR / 'learning_curve.png'}")

    _write_experiment_log(args, agent)


def _write_experiment_log(args: argparse.Namespace, agent: DQNAgent) -> None:
    """Append run summary to notes/experiments.md per CLAUDE.md requirement."""
    log_path = pathlib.Path("notes/experiments.md")
    log_path.parent.mkdir(exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"""
## DQN run — {ts}

**Config**
- total_steps: {args.total_steps:,}
- warmup: {args.warmup:,}
- buffer_cap: {args.buffer_cap:,}
- batch_size: {args.batch_size}
- grad_freq: every {args.grad_freq} env steps
- n_envs: {args.n_envs}
- device: {agent.device}
- network: SuikaQNetwork (per-fruit MLP 13→128→128, mean+max pool, head 266→256→256→32)
- algorithm: Double DQN, γ=0.99, Adam lr=3e-4 cosine→1e-5, grad_clip=10, target_sync=1000 grad steps
- reward: raw (no scaling), Huber loss
- epsilon: 1.0→0.05 over 100k env steps

**Results**
*(fill in after run: final mean return, game score, steps/ep, convergence step)*

**What changed vs. prior runs**
*(fill in)*
"""
    with open(log_path, "a") as f:
        f.write(entry)
    print(f"Experiment log appended → {log_path}")


if __name__ == "__main__":
    main()
