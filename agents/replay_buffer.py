# Generated with Claude Code (claude-sonnet-4-6). Architecture directed by David Huang.
# Circular replay buffer with compressed storage: fruit types stored as int8 index,
# expanded to one-hot only at sample time.

import numpy as np
from typing import Dict

from suika_env.constants import MAX_FRUITS, NUM_FRUIT_TYPES

_NUM_FRUIT_TYPES = NUM_FRUIT_TYPES


class DictReplayBuffer:
    """
    Circular replay buffer for Dict observations from SuikaEnv.

    Compression: fruit_types stored as int8 index (0-10) rather than float32
    one-hot, reducing per-transition memory ~10x for that field.
    At 100k capacity: ~120MB total.

    Interface:
        push(obs, action, reward, next_obs, done)
        sample(batch_size) -> dict with 'obs', 'next_obs', 'actions', 'rewards', 'dones'
    """

    def __init__(self, capacity: int = 100_000, max_fruits: int = MAX_FRUITS) -> None:
        self.capacity  = capacity
        self.max_fruits = max_fruits
        self._pos  = 0
        self._size = 0

        # Current observation fields
        self._s_fruits  = np.zeros((capacity, max_fruits, 3), dtype=np.float32)
        self._s_typeidx = np.zeros((capacity, max_fruits),    dtype=np.int8)
        self._s_mask    = np.zeros((capacity, max_fruits),    dtype=np.int8)
        self._s_cur     = np.zeros(capacity,                  dtype=np.int8)
        self._s_nxt     = np.zeros(capacity,                  dtype=np.int8)

        # Next observation fields
        self._ns_fruits  = np.zeros((capacity, max_fruits, 3), dtype=np.float32)
        self._ns_typeidx = np.zeros((capacity, max_fruits),    dtype=np.int8)
        self._ns_mask    = np.zeros((capacity, max_fruits),    dtype=np.int8)
        self._ns_cur     = np.zeros(capacity,                  dtype=np.int8)
        self._ns_nxt     = np.zeros(capacity,                  dtype=np.int8)

        # Transition fields
        self._actions = np.zeros(capacity, dtype=np.int8)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones   = np.zeros(capacity, dtype=bool)

    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    def push(
        self,
        obs:      Dict,
        action:   int,
        reward:   float,
        next_obs: Dict,
        done:     bool,
    ) -> None:
        i = self._pos
        self._s_fruits[i]  = obs["fruits"]
        self._s_typeidx[i] = obs["fruit_types"].argmax(axis=-1).astype(np.int8)
        self._s_mask[i]    = obs["fruit_mask"]
        self._s_cur[i]     = obs["current_fruit"]
        self._s_nxt[i]     = obs["next_fruit"]

        self._ns_fruits[i]  = next_obs["fruits"]
        self._ns_typeidx[i] = next_obs["fruit_types"].argmax(axis=-1).astype(np.int8)
        self._ns_mask[i]    = next_obs["fruit_mask"]
        self._ns_cur[i]     = next_obs["current_fruit"]
        self._ns_nxt[i]     = next_obs["next_fruit"]

        self._actions[i] = action
        self._rewards[i] = reward
        self._dones[i]   = done

        self._pos  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict:
        assert self._size >= batch_size, "Buffer too small to sample requested batch."
        idxs = np.random.randint(0, self._size, size=batch_size)

        eye = np.eye(_NUM_FRUIT_TYPES, dtype=np.float32)
        s_type_oh  = eye[self._s_typeidx[idxs].astype(np.int64)]   # [B, 100, 11]
        ns_type_oh = eye[self._ns_typeidx[idxs].astype(np.int64)]  # [B, 100, 11]

        return {
            "obs": {
                "fruits":        self._s_fruits[idxs],
                "fruit_types":   s_type_oh,
                "fruit_mask":    self._s_mask[idxs],
                "current_fruit": self._s_cur[idxs].astype(np.int64),
                "next_fruit":    self._s_nxt[idxs].astype(np.int64),
            },
            "next_obs": {
                "fruits":        self._ns_fruits[idxs],
                "fruit_types":   ns_type_oh,
                "fruit_mask":    self._ns_mask[idxs],
                "current_fruit": self._ns_cur[idxs].astype(np.int64),
                "next_fruit":    self._ns_nxt[idxs].astype(np.int64),
            },
            "actions": self._actions[idxs].astype(np.int64),
            "rewards": self._rewards[idxs],
            "dones":   self._dones[idxs],
        }
