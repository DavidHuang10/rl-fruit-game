# AI-assisted random baseline; included to anchor learned-agent performance.
from __future__ import annotations


class RandomAgent:
    """Uniform-random baseline for discrete-action environments."""

    def __init__(self, action_space) -> None:
        self.action_space = action_space

    def select_action(self, obs: dict | None = None, eval_mode: bool = False) -> int:
        return int(self.action_space.sample())
