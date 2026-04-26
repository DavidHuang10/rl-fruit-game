from __future__ import annotations


class CenterAgent:
    """Deterministic baseline that always drops in the center action bin."""

    def __init__(self, action_space) -> None:
        self.action = int(action_space.n // 2)

    def select_action(self, obs: dict | None = None, eval_mode: bool = False) -> int:
        return self.action
