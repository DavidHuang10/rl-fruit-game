from __future__ import annotations

import argparse
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from suika_env import SuikaEnv


RESULTS_DIR = pathlib.Path("results/ppo")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total_timesteps", type=int, default=300_000)
    p.add_argument("--log_freq", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_steps", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--n_envs", type=int, default=4)
    return p.parse_args()


class MetricsCallback(BaseCallback):
    def __init__(self, metrics_path: pathlib.Path, log_freq: int) -> None:
        super().__init__()
        self.metrics_path = metrics_path
        self.log_freq = log_freq
        self.ep_returns: list[float] = []
        self.ep_scores: list[float] = []
        self.ep_lengths: list[int] = []
        self._csv_file = None
        self._writer = None
        self._last_log_step = 0

    def _on_training_start(self) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = open(self.metrics_path, "w", newline="")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "env_step",
                "mean_ep_return",
                "mean_ep_score",
                "mean_ep_length",
            ],
        )
        self._writer.writeheader()
        self._csv_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            episode = info.get("episode")
            if episode is not None:
                self.ep_returns.append(float(episode["r"]))
                self.ep_lengths.append(int(episode["l"]))
                self.ep_scores.append(float(info.get("score", episode["r"])))

        if self.num_timesteps - self._last_log_step >= self.log_freq:
            self._write_row()
            self._last_log_step = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        self._write_row()
        if self._csv_file is not None:
            self._csv_file.close()

    def _write_row(self) -> None:
        assert self._writer is not None
        assert self._csv_file is not None
        returns = self.ep_returns[-100:]
        scores = self.ep_scores[-100:]
        lengths = self.ep_lengths[-100:]
        row = {
            "env_step": self.num_timesteps,
            "mean_ep_return": f"{np.mean(returns):.2f}" if returns else "nan",
            "mean_ep_score": f"{np.mean(scores):.2f}" if scores else "nan",
            "mean_ep_length": f"{np.mean(lengths):.1f}" if lengths else "nan",
        }
        self._writer.writerow(row)
        self._csv_file.flush()
        print(
            f"step {self.num_timesteps:>8,} | "
            f"ret {row['mean_ep_return']:>8} | "
            f"score {row['mean_ep_score']:>8} | "
            f"len {row['mean_ep_length']:>6}"
        )


def _save_curves(metrics_path: pathlib.Path) -> None:
    env_steps, ep_returns, ep_lengths = [], [], []
    with open(metrics_path) as f:
        for row in csv.DictReader(f):
            env_steps.append(int(row["env_step"]))
            ep_returns.append(float(row["mean_ep_return"]))
            ep_lengths.append(float(row["mean_ep_length"]))

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(env_steps, ep_returns, linewidth=1)
    axes[0].set_ylabel("Mean episode return (last 100 eps)")
    axes[0].set_title("PPO - Suika training curves")
    axes[1].plot(env_steps, ep_lengths, linewidth=1, color="orange")
    axes[1].set_ylabel("Mean episode length (last 100 eps)")
    axes[1].set_xlabel("Env steps")
    fig.tight_layout()
    fig.savefig(metrics_path.parent / "learning_curve.png", dpi=150)
    plt.close(fig)


def make_env(seed: int):
    def _init():
        env = SuikaEnv()
        env.reset(seed=seed)
        return Monitor(env, info_keywords=("score", "num_fruits"))

    return _init


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    env_fns = [make_env(args.seed + i) for i in range(args.n_envs)]
    env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)
    metrics_path = RESULTS_DIR / "metrics.csv"
    callback = MetricsCallback(metrics_path=metrics_path, log_freq=args.log_freq)

    model = PPO(
        "MultiInputPolicy",
        env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        seed=args.seed,
        verbose=1,
    )

    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"Envs:            {args.n_envs}")
    print(f"Rollout batch:   {args.n_steps * args.n_envs:,} transitions")
    print(f"Results:         {RESULTS_DIR.resolve()}")
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    model.save(RESULTS_DIR / "model")
    env.close()
    _save_curves(metrics_path)
    print(f"Done. Model -> {RESULTS_DIR / 'model.zip'}")
    print(f"Curves -> {RESULTS_DIR / 'learning_curve.png'}")


if __name__ == "__main__":
    main()
