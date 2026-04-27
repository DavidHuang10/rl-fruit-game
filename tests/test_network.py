# AI-assisted network tests; shape and masking cases chosen to validate the DQN architecture.

import numpy as np
import pytest
import torch

from agents.network import (
    SuikaQNetwork,
    obs_to_tensor,
    N_ACTIONS,
    MAX_FRUITS,
    NUM_FRUIT_TYPES,
)


def _make_obs(n_active: int = 5, batch: bool = False, B: int = 4):
    """Synthesise a realistic Dict obs (single or batched)."""

    def _single():
        fruits = np.zeros((MAX_FRUITS, 4), dtype=np.float32)
        types = np.zeros((MAX_FRUITS, NUM_FRUIT_TYPES), dtype=np.float32)
        mask = np.zeros(MAX_FRUITS, dtype=np.int8)
        for i in range(n_active):
            fruits[i, 0] = np.random.uniform(0.1, 0.9)  # x
            fruits[i, 1] = np.random.uniform(0.1, 0.9)  # y
            t = np.random.randint(0, NUM_FRUIT_TYPES)
            types[i, t] = 1.0
            mask[i] = 1
        return {
            "fruits": fruits,
            "fruit_types": types,
            "fruit_mask": mask,
            "current_fruit": np.int64(np.random.randint(0, 5)),
            "next_fruit": np.int64(np.random.randint(0, 5)),
        }

    if not batch:
        return _single()
    obs_list = [_single() for _ in range(B)]
    return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}


@pytest.fixture
def net():
    return SuikaQNetwork()


@pytest.fixture
def device():
    return torch.device("cpu")


# ── forward pass ──────────────────────────────────────────────────────────────


def test_forward_output_shape(net, device):
    obs = _make_obs(n_active=7, batch=True, B=4)
    t = obs_to_tensor(obs, device)
    q = net(**t)
    assert q.shape == (4, N_ACTIONS)


def test_forward_single_obs(net, device):
    obs = _make_obs(n_active=3)
    t = obs_to_tensor(obs, device)
    q = net(**t)
    assert q.shape == (1, N_ACTIONS)


def test_forward_no_nan(net, device):
    obs = _make_obs(n_active=10, batch=True, B=8)
    t = obs_to_tensor(obs, device)
    q = net(**t)
    assert not torch.isnan(q).any()
    assert not torch.isinf(q).any()


# ── obs_to_tensor: velocities are dropped ────────────────────────────────────


def test_velocities_dropped(device):
    obs = _make_obs(n_active=5)
    t = obs_to_tensor(obs, device)
    assert t["fruits"].shape[-1] == 2, "vx/vy should be dropped; only x,y remain"


# ── padding invariance ────────────────────────────────────────────────────────


def test_padding_invariance(net, device):
    """Same active fruits in different slot positions must yield identical Q-values."""
    n_active = 6
    fruits = np.random.rand(n_active, 4).astype(np.float32)
    types = np.eye(NUM_FRUIT_TYPES, dtype=np.float32)[
        np.random.randint(0, NUM_FRUIT_TYPES, n_active)
    ]

    # Place active fruits in slots 0..5
    def _make_with_offset(offset):
        o = {
            "fruits": np.zeros((MAX_FRUITS, 4), dtype=np.float32),
            "fruit_types": np.zeros((MAX_FRUITS, NUM_FRUIT_TYPES), dtype=np.float32),
            "fruit_mask": np.zeros(MAX_FRUITS, dtype=np.int8),
            "current_fruit": np.int64(2),
            "next_fruit": np.int64(1),
        }
        for i in range(n_active):
            o["fruits"][offset + i] = fruits[i]
            o["fruit_types"][offset + i] = types[i]
            o["fruit_mask"][offset + i] = 1
        return o

    obs_early = _make_with_offset(0)
    obs_late = _make_with_offset(MAX_FRUITS - n_active)

    net.eval()
    with torch.no_grad():
        q_early = net(**obs_to_tensor(obs_early, device))
        q_late = net(**obs_to_tensor(obs_late, device))

    assert torch.allclose(
        q_early, q_late, atol=1e-5
    ), "Q-values should be identical regardless of where active fruits sit in the slot array"


# ── replay buffer input roundtrip ─────────────────────────────────────────────


def test_obs_to_tensor_batched_matches_stacked_singles(net, device):
    """Batching N single obs should give the same result as calling obs_to_tensor on the batch."""
    singles = [_make_obs(n_active=i + 1) for i in range(4)]
    batched = {k: np.stack([s[k] for s in singles]) for k in singles[0]}

    net.eval()
    with torch.no_grad():
        q_batch = net(**obs_to_tensor(batched, device))
        q_stack = torch.cat([net(**obs_to_tensor(s, device)) for s in singles], dim=0)

    assert torch.allclose(q_batch, q_stack, atol=1e-5)
