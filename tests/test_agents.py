from __future__ import annotations

import numpy as np
from gymnasium import spaces

import agents.factory as factory
import agents.ppo as ppo_module
from agents import CenterAgent, DQNAgent, DictReplayBuffer, PPOAgent, RandomAgent, SuikaQNetwork
from suika_env.constants import ACTION_COUNT, MAX_FRUITS, NUM_FRUIT_TYPES, SPAWN_POOL_SIZE
from suika_env.env import SuikaEnv


def test_agents_package_exports_public_agent_types():
    assert DQNAgent is not None
    assert PPOAgent is not None
    assert RandomAgent is not None
    assert CenterAgent is not None
    assert DictReplayBuffer is not None
    assert SuikaQNetwork is not None


def test_ppo_agent_uses_sb3_predict_interface(monkeypatch):
    calls = {}

    class FakeModel:
        def predict(self, obs, deterministic):
            calls["obs"] = obs
            calls["deterministic"] = deterministic
            return np.int64(7), None

    def fake_load(checkpoint):
        calls["checkpoint"] = checkpoint
        return FakeModel()

    monkeypatch.setattr(ppo_module.PPO, "load", fake_load)

    obs = {"current_fruit": np.int64(1)}
    agent = PPOAgent("checkpoint.zip")

    assert agent.select_action(obs, eval_mode=True) == 7
    assert calls == {
        "checkpoint": "checkpoint.zip",
        "obs": obs,
        "deterministic": True,
    }


def test_agent_factory_builds_simple_baselines():
    action_space = spaces.Discrete(ACTION_COUNT)

    random_agent = factory.build_agent("random", action_space)
    center_agent = factory.build_agent("center", action_space)

    assert isinstance(random_agent, RandomAgent)
    assert isinstance(center_agent, CenterAgent)
    assert center_agent.select_action({}) == ACTION_COUNT // 2


def test_agent_factory_uses_default_checkpoints(monkeypatch):
    loaded = {}

    class FakeDQNAgent:
        def load(self, checkpoint):
            loaded["dqn"] = checkpoint

        def select_action(self, obs, eval_mode=True):
            return 0

    class FakePPOAgent:
        def __init__(self, checkpoint):
            loaded["ppo"] = checkpoint

        def select_action(self, obs, eval_mode=True):
            return 0

    monkeypatch.setattr(factory, "DQNAgent", FakeDQNAgent)
    monkeypatch.setattr(factory, "PPOAgent", FakePPOAgent)

    action_space = spaces.Discrete(ACTION_COUNT)
    factory.build_agent("dqn", action_space)
    factory.build_agent("ppo", action_space)

    assert loaded["dqn"] == factory.DQN_DEFAULT_CHECKPOINT
    assert loaded["ppo"] == factory.PPO_DEFAULT_CHECKPOINT


def test_shared_constants_match_default_env_spaces():
    env = SuikaEnv()
    try:
        assert env.action_space.n == ACTION_COUNT
        assert env.observation_space["fruits"].shape == (MAX_FRUITS, 3)
        assert env.observation_space["fruit_types"].shape == (MAX_FRUITS, NUM_FRUIT_TYPES)
        assert env.observation_space["current_fruit"].n == SPAWN_POOL_SIZE
        assert env.observation_space["next_fruit"].n == SPAWN_POOL_SIZE
    finally:
        env.close()
