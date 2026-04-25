from __future__ import annotations

import pathlib

from stable_baselines3 import PPO


class PPOAgent:
    """Stable-Baselines3 PPO policy wrapper with the shared agent interface."""

    def __init__(self, checkpoint: str | pathlib.Path) -> None:
        self.model = PPO.load(checkpoint)

    def select_action(self, obs: dict, eval_mode: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=eval_mode)
        return int(action)
