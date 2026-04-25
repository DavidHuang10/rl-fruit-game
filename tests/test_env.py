"""Gymnasium API compliance and integration tests for SuikaEnv."""
import numpy as np
import pytest
import gymnasium as gym

import suika_env  # registers suika_env/SuikaEnv-v0
from suika_env.env import SuikaEnv
from suika_env.config import EnvConfig


# ---------------------------------------------------------------------------
# Construction and registration
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_gym_make(self):
        env = gym.make("suika_env/SuikaEnv-v0")
        assert env is not None
        env.close()

    def test_direct_instantiation(self):
        env = SuikaEnv()
        assert env is not None
        env.close()

    def test_custom_config(self):
        cfg = EnvConfig(n_action_bins=16, max_fruits=50)
        env = SuikaEnv(config=cfg)
        assert env.action_space.n == 16
        env.close()

    def test_pixel_mode_raises(self):
        cfg = EnvConfig(observation_mode="pixels")
        with pytest.raises(NotImplementedError):
            SuikaEnv(config=cfg)


# ---------------------------------------------------------------------------
# Gymnasium API compliance
# ---------------------------------------------------------------------------

class TestGymnasiumAPI:
    def test_check_env(self):
        from stable_baselines3.common.env_checker import check_env
        env = SuikaEnv()
        check_env(env, warn=True)
        env.close()

    def test_obs_after_reset_in_space(self):
        env = SuikaEnv()
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs), "Obs after reset is outside obs space"
        env.close()

    def test_obs_after_step_in_space(self):
        env = SuikaEnv()
        env.reset(seed=7)
        for _ in range(20):
            obs, _, terminated, _, _ = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs)
            if terminated:
                break
        env.close()

    def test_action_space_sample_valid(self):
        env = SuikaEnv()
        env.reset(seed=1)
        for _ in range(50):
            action = env.action_space.sample()
            assert env.action_space.contains(action)
        env.close()

    def test_edge_actions(self):
        env = SuikaEnv()
        env.reset(seed=2)
        n = env.cfg.n_action_bins
        obs, r, terminated, truncated, info = env.step(0)
        assert env.observation_space.contains(obs)
        if not terminated:
            env.step(n - 1)
        env.close()

    def test_episode_terminates(self):
        env = SuikaEnv()
        env.reset(seed=3)
        terminated = False
        for _ in range(500):
            _, _, terminated, _, _ = env.step(env.action_space.sample())
            if terminated:
                break
        assert terminated, "Episode never terminated in 500 steps"
        env.close()

    def test_info_keys(self):
        env = SuikaEnv()
        _, info = env.reset()
        assert "score" in info
        _, _, _, _, step_info = env.step(0)
        assert "score" in step_info
        assert "physics_frames" in step_info
        assert "num_fruits" in step_info
        env.close()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_fruit_sequence(self):
        def run_episode(seed):
            env = SuikaEnv()
            obs, _ = env.reset(seed=seed)
            history = [obs["current_fruit"], obs["next_fruit"]]
            for _ in range(10):
                obs, _, terminated, _, _ = env.step(0)
                history.append(obs["current_fruit"])
                if terminated:
                    break
            env.close()
            return history

        h1 = run_episode(42)
        h2 = run_episode(42)
        assert h1 == h2, "Same seed produced different fruit sequences"

    def test_different_seeds_differ(self):
        def fruit_seq(seed):
            env = SuikaEnv()
            env.reset(seed=seed)
            seq = []
            for _ in range(20):
                obs, _, terminated, _, _ = env.step(0)
                seq.append(obs["current_fruit"])
                if terminated:
                    break
            env.close()
            return seq

        # Very unlikely to be identical for different seeds
        assert fruit_seq(0) != fruit_seq(99)


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------

class TestRewards:
    def test_reward_nonnegative(self):
        env = SuikaEnv()
        env.reset(seed=10)
        for _ in range(30):
            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            assert reward >= 0, f"Negative reward {reward}"
            if terminated:
                break
        env.close()

    def test_score_equals_sum_of_rewards(self):
        env = SuikaEnv()
        env.reset(seed=11)
        total = 0.0
        for _ in range(50):
            _, reward, terminated, _, info = env.step(env.action_space.sample())
            total += reward
            if terminated:
                assert abs(total - info["score"]) < 1e-3
                break
        env.close()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:
    def test_rgb_array_shape(self):
        env = SuikaEnv(render_mode="rgb_array")
        env.reset(seed=0)
        frame = env.render()
        assert frame is not None
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        env.close()
