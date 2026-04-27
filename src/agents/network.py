# AI-assisted Q-network; I decided on the DeepSets-style design for unordered fruit states as well as the parameters to use for hidden layers, activation, etc.
# DeepSets-style Q-network: per-fruit MLP -> masked mean+max pool -> head MLP -> Q[32].

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from suika_env.constants import (
    ACTION_COUNT,
    MAX_FRUITS,
    NUM_FRUIT_TYPES,
    SPAWN_POOL_SIZE,
)

POOL_SIZE = SPAWN_POOL_SIZE  # spawnable tiers 0-4
N_ACTIONS = ACTION_COUNT
PER_FRUIT_IN = 13  # x, y (2) + one-hot type (11)


def obs_to_tensor(obs: Dict, device: torch.device) -> Dict[str, torch.Tensor]:
    """Convert a numpy Dict obs to tensors.

    Human-edited: I kept this conversion aligned with the structured
    observation design rather than flattening the board into one vector.
    """
    fruits_np = np.asarray(obs["fruits"])
    single = fruits_np.ndim == 2

    def prep(arr, add_batch):
        a = np.asarray(arr)
        return a[None] if add_batch else a

    fruits_np = prep(obs["fruits"], single)  # [B, 100, 4] or [1, 100, 4]
    types_np = prep(obs["fruit_types"], single)  # [B, 100, 11]
    mask_np = prep(obs["fruit_mask"], single)  # [B, 100]
    cur_np = prep(obs["current_fruit"], single)  # [B] or [1]
    nxt_np = prep(obs["next_fruit"], single)  # [B] or [1]

    return {
        # drop vx, vy (channels 2 and 3) — always ~0 at every settled obs
        "fruits": torch.tensor(fruits_np[:, :, :2], dtype=torch.float32, device=device),
        "fruit_types": torch.tensor(types_np, dtype=torch.float32, device=device),
        "fruit_mask": torch.tensor(
            mask_np.astype(np.float32), dtype=torch.float32, device=device
        ),
        "current_fruit": torch.tensor(
            cur_np.astype(np.int64), dtype=torch.long, device=device
        ),
        "next_fruit": torch.tensor(
            nxt_np.astype(np.int64), dtype=torch.long, device=device
        ),
    }


class SuikaQNetwork(nn.Module):
    """
    Q-network for Suika using set-aggregation over the variable-length fruit set.

    Architecture:
      per-fruit MLP (13 -> 128 -> 128)
      -> masked mean + max pool -> [256]
      -> concat one-hot(current_fruit, 5) + one-hot(next_fruit, 5) -> [266]
      -> head MLP (266 -> 256 -> 256 -> 32)
      -> Q-values [32]

    NOTE: I chose this set-aggregation architecture because active
    fruits have no meaningful fixed ordering on the board. I also decided
    the parameters like hidden layer size, and activation function.
    """

    def __init__(
        self,
        per_fruit_hidden: int = 128,
        head_hidden: int = 256,
        n_actions: int = N_ACTIONS,
        n_fruit_types: int = NUM_FRUIT_TYPES,
        n_spawn_types: int = POOL_SIZE,
    ) -> None:
        super().__init__()
        self._n_spawn = n_spawn_types
        per_fruit_in = 2 + n_fruit_types  # x, y + one-hot type

        self.fruit_encoder = nn.Sequential(
            nn.Linear(per_fruit_in, per_fruit_hidden),
            nn.ReLU(),
            nn.Linear(per_fruit_hidden, per_fruit_hidden),
            nn.ReLU(),
        )

        set_dim = per_fruit_hidden * 2  # mean + max concat
        cat_dim = n_spawn_types * 2  # current + next one-hots
        head_in = set_dim + cat_dim  # 266

        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, n_actions),
        )

    def forward(
        self,
        fruits: torch.Tensor,  # [B, 100, 2]   x/W, y/H
        fruit_types: torch.Tensor,  # [B, 100, 11]  one-hot
        fruit_mask: torch.Tensor,  # [B, 100]      float32, 1=active 0=padding
        current_fruit: torch.Tensor,  # [B]           long
        next_fruit: torch.Tensor,  # [B]           long
    ) -> torch.Tensor:  # [B, 32]
        # Per-fruit features: concat spatial + type -> [B, 100, 13]
        per_fruit = torch.cat([fruits, fruit_types], dim=-1)

        # Shared per-fruit MLP -> [B, 100, 128]
        encoded = self.fruit_encoder(per_fruit)

        # ── masked mean pool ──────────────────────────────────────────────
        mask = fruit_mask.unsqueeze(-1)  # [B, 100, 1]
        sum_active = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
        mean_pool = (encoded * mask).sum(dim=1) / sum_active  # [B, 128]

        # ── masked max pool ───────────────────────────────────────────────
        max_input = encoded.masked_fill(mask == 0.0, float("-inf"))  # [B, 100, 128]
        max_pool, _ = max_input.max(dim=1)  # [B, 128]
        # Guard against the degenerate all-masked case (shouldn't occur in Suika).
        any_active = fruit_mask.any(dim=1, keepdim=True)  # [B, 1]
        max_pool = torch.where(any_active, max_pool, torch.zeros_like(max_pool))

        # ── categorical context ───────────────────────────────────────────
        cur_oh = F.one_hot(current_fruit, self._n_spawn).float()  # [B, 5]
        nxt_oh = F.one_hot(next_fruit, self._n_spawn).float()  # [B, 5]

        # ── assemble state vector and head ────────────────────────────────
        set_summary = torch.cat([mean_pool, max_pool], dim=-1)  # [B, 256]
        state = torch.cat([set_summary, cur_oh, nxt_oh], dim=-1)  # [B, 266]
        return self.head(state)  # [B, 32]
