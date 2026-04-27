# AI-assisted replay-buffer tests; sampling and compact storage behavior reviewed by me.

import numpy as np
import pytest

from agents.replay_buffer import DictReplayBuffer, _NUM_FRUIT_TYPES

MAX_FRUITS = 100


def _make_obs(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_active = rng.integers(1, 15)
    fruits = np.zeros((MAX_FRUITS, 4), dtype=np.float32)
    types = np.zeros((MAX_FRUITS, _NUM_FRUIT_TYPES), dtype=np.float32)
    mask = np.zeros(MAX_FRUITS, dtype=np.int8)
    for i in range(n_active):
        fruits[i, :2] = rng.uniform(0.1, 0.9, 2)
        t = rng.integers(0, _NUM_FRUIT_TYPES)
        types[i, t] = 1.0
        mask[i] = 1
    return {
        "fruits": fruits,
        "fruit_types": types,
        "fruit_mask": mask,
        "current_fruit": np.int64(rng.integers(0, 5)),
        "next_fruit": np.int64(rng.integers(0, 5)),
    }


@pytest.fixture
def buf():
    return DictReplayBuffer(capacity=100)


def _fill(buf: DictReplayBuffer, n: int) -> None:
    for i in range(n):
        buf.push(
            _make_obs(i),
            action=i % 32,
            reward=float(i),
            next_obs=_make_obs(i + 1000),
            done=(i % 10 == 9),
        )


# ── basic size tracking ───────────────────────────────────────────────────────


def test_size_grows(buf):
    assert buf.size == 0
    _fill(buf, 10)
    assert buf.size == 10


def test_size_caps_at_capacity(buf):
    _fill(buf, 150)
    assert buf.size == 100


# ── push/sample roundtrip ─────────────────────────────────────────────────────


def test_sample_shapes(buf):
    _fill(buf, 50)
    batch = buf.sample(16)
    assert batch["obs"]["fruits"].shape == (16, MAX_FRUITS, 4)
    assert batch["obs"]["fruit_types"].shape == (16, MAX_FRUITS, _NUM_FRUIT_TYPES)
    assert batch["obs"]["fruit_mask"].shape == (16, MAX_FRUITS)
    assert batch["obs"]["current_fruit"].shape == (16,)
    assert batch["obs"]["next_fruit"].shape == (16,)
    assert batch["next_obs"]["fruits"].shape == (16, MAX_FRUITS, 4)
    assert batch["actions"].shape == (16,)
    assert batch["rewards"].shape == (16,)
    assert batch["dones"].shape == (16,)


def test_sample_dtypes(buf):
    _fill(buf, 50)
    batch = buf.sample(8)
    assert batch["obs"]["fruits"].dtype == np.float32
    assert batch["obs"]["fruit_types"].dtype == np.float32
    assert batch["rewards"].dtype == np.float32
    assert batch["actions"].dtype == np.int64
    assert batch["obs"]["current_fruit"].dtype == np.int64


def test_fruit_types_are_valid_one_hot(buf):
    _fill(buf, 50)
    batch = buf.sample(16)
    types = batch["obs"]["fruit_types"]
    # Every row should sum to 1 or 0 (0 for masked-out slots)
    row_sums = types.sum(axis=-1)
    assert np.all((row_sums == 0) | (row_sums == 1))


def test_rewards_roundtrip():
    """Rewards pushed should be retrievable (sampling all)."""
    small = DictReplayBuffer(capacity=10)
    for i in range(10):
        small.push(
            _make_obs(i),
            action=0,
            reward=float(i * 10),
            next_obs=_make_obs(i),
            done=False,
        )
    batch = small.sample(10)
    assert set(batch["rewards"].astype(int).tolist()).issubset(set(range(0, 100, 10)))


def test_circular_overwrite(buf):
    """After overwrite, old data is gone, new data is present."""
    _fill(buf, 100)
    # Now push 10 more — overwrites first 10 slots
    for i in range(10):
        buf.push(
            _make_obs(200 + i),
            action=0,
            reward=999.0,
            next_obs=_make_obs(300 + i),
            done=False,
        )
    assert buf.size == 100  # still full
    # reward 999 should exist in the buffer somewhere
    all_rewards = buf._rewards[: buf.size]
    assert (all_rewards == 999.0).sum() == 10


def test_sample_requires_enough_data(buf):
    _fill(buf, 5)
    with pytest.raises(AssertionError):
        buf.sample(10)
