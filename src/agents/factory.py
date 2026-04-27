# AI-assisted agent factory; final checkpoint paths and shared select_action interface reviewed by me.
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from gymnasium import spaces

from agents.center_agent import CenterAgent
from agents.dqn import DQNAgent
from agents.ppo import PPOAgent
from agents.random_agent import RandomAgent

DQN_DEFAULT_CHECKPOINT = Path("models/dqn/model.pt")
PPO_DEFAULT_CHECKPOINT = Path("models/ppo/model.zip")


class SelectsAction(Protocol):
    def select_action(self, obs: dict, eval_mode: bool = True) -> int: ...


def build_agent(
    kind: str,
    action_space: spaces.Discrete,
    checkpoint: str | Path | None = None,
) -> SelectsAction:
    if kind == "random":
        return RandomAgent(action_space)
    if kind == "center":
        return CenterAgent(action_space)
    if kind == "dqn":
        agent = DQNAgent()
        agent.load(checkpoint or DQN_DEFAULT_CHECKPOINT)
        return agent
    if kind == "ppo":
        return PPOAgent(checkpoint or PPO_DEFAULT_CHECKPOINT)
    raise ValueError(f"Unknown agent: {kind}")
