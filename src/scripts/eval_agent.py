# AI-assisted evaluation script;
from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np

from agents import SelectsAction, build_agent
from suika_env import SuikaEnv


RESULTS_DIR = pathlib.Path("results/eval")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", choices=["random", "center", "dqn", "ppo"], required=True)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out_dir", type=pathlib.Path, default=RESULTS_DIR)
    return p.parse_args()


def evaluate(agent: SelectsAction, episodes: int, seed: int) -> list[dict[str, float]]:
    """Run complete evaluation episodes and collect per-episode metrics.

    Human-edited: I chose these metrics to compare both score and board health,
    not just raw return.
    """
    rows: list[dict[str, float]] = []
    env = SuikaEnv()
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_return = 0.0
            ep_len = 0
            final_info = {"score": 0, "num_fruits": 0, "physics_frames": 0}
            while True:
                action = agent.select_action(obs, eval_mode=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                ep_len += 1
                final_info = info
                if terminated or truncated:
                    break

            rows.append(
                {
                    "episode": float(ep),
                    "episode_return": ep_return,
                    "final_score": float(final_info.get("score", ep_return)),
                    "episode_length": float(ep_len),
                    "final_num_fruits": float(final_info.get("num_fruits", 0)),
                    "last_physics_frames": float(final_info.get("physics_frames", 0)),
                }
            )
    finally:
        env.close()
    return rows


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = [
        "episode_return",
        "final_score",
        "episode_length",
        "final_num_fruits",
        "last_physics_frames",
    ]
    summary: dict[str, float] = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std(ddof=0))
    return summary


def save_outputs(
    agent_name: str,
    rows: list[dict[str, float]],
    summary: dict[str, float],
    out_dir: pathlib.Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{agent_name}_episodes.csv"
    json_path = out_dir / f"{agent_name}_summary.json"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)


def main() -> None:
    args = parse_args()
    env = SuikaEnv()
    try:
        agent = build_agent(args.agent, env.action_space, args.checkpoint)
    finally:
        env.close()

    rows = evaluate(agent, episodes=args.episodes, seed=args.seed)
    summary = summarize(rows)
    save_outputs(args.agent, rows, summary, args.out_dir)

    print(f"Agent: {args.agent}")
    print(f"Episodes: {args.episodes}")
    print(
        f"Mean return: {summary['mean_episode_return']:.2f} +/- {summary['std_episode_return']:.2f}"
    )
    print(
        f"Mean score:  {summary['mean_final_score']:.2f} +/- {summary['std_final_score']:.2f}"
    )
    print(
        f"Mean length: {summary['mean_episode_length']:.1f} +/- {summary['std_episode_length']:.1f}"
    )
    print(
        f"Mean fruits: {summary['mean_final_num_fruits']:.1f} +/- {summary['std_final_num_fruits']:.1f}"
    )
    print(f"Saved: {args.out_dir / f'{args.agent}_episodes.csv'}")
    print(f"Saved: {args.out_dir / f'{args.agent}_summary.json'}")


if __name__ == "__main__":
    main()
