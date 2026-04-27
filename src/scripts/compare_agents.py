# AI-assisted plotting script; comparison metrics selected by me.
"""Load per-episode CSVs for all four agents and produce a comparison bar chart with error bars."""
from __future__ import annotations

import json
import pathlib

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np

EVAL_DIR = pathlib.Path("results/eval")
OUT_DIR = pathlib.Path("results/eval")

AGENTS = ["random", "center", "ppo", "dqn"]
LABELS = ["Random", "Center", "PPO", "DQN"]
COLORS = ["#888888", "#4C72B0", "#DD8452", "#55A868"]

METRICS = {
    "mean_episode_return": ("Mean Episode Return", "Return"),
    "mean_episode_length": ("Mean Episode Length", "Steps"),
    "mean_final_num_fruits": ("Mean Final Fruit Count", "Fruits"),
}


def load_summaries() -> dict[str, dict[str, float]]:
    out = {}
    for agent in AGENTS:
        path = EVAL_DIR / f"{agent}_summary.json"
        with open(path) as f:
            out[agent] = json.load(f)
    return out


def bar_chart(
    summaries: dict[str, dict[str, float]],
    metric_key: str,
    title: str,
    ylabel: str,
    ax: matplotlib.axes.Axes,
) -> None:
    means = [summaries[a][metric_key] for a in AGENTS]
    std_key = metric_key.replace("mean_", "std_")
    stds = [summaries[a].get(std_key, 0.0) for a in AGENTS]
    x = np.arange(len(AGENTS))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=COLORS, width=0.55, zorder=3)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for bar, mean in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(stds) * 0.05,
            f"{mean:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main() -> None:
    summaries = load_summaries()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        "Agent Comparison — 50 Episodes Each", fontsize=13, fontweight="bold", y=1.01
    )

    for ax, (key, (title, ylabel)) in zip(axes, METRICS.items()):
        bar_chart(summaries, key, title, ylabel, ax)

    fig.tight_layout()
    out_path = OUT_DIR / "comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
