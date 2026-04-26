from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from suika_env.constants import NUM_FRUIT_TYPES, SPAWN_POOL_SIZE


class SuikaPPOFeaturesExtractor(BaseFeaturesExtractor):
    """Object-aware PPO feature extractor for Suika dict observations."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 266,
        per_fruit_hidden: int = 128,
        n_fruit_types: int = NUM_FRUIT_TYPES,
        n_spawn_types: int = SPAWN_POOL_SIZE,
    ) -> None:
        super().__init__(observation_space, features_dim)
        self.n_spawn_types = n_spawn_types

        per_fruit_in = 3 + n_fruit_types
        self.fruit_encoder = nn.Sequential(
            nn.Linear(per_fruit_in, per_fruit_hidden),
            nn.ReLU(),
            nn.Linear(per_fruit_hidden, per_fruit_hidden),
            nn.ReLU(),
        )

        expected_features_dim = per_fruit_hidden * 2 + n_spawn_types * 2
        if features_dim != expected_features_dim:
            raise ValueError(
                f"features_dim must be {expected_features_dim} for this extractor, "
                f"got {features_dim}"
            )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        fruits = observations["fruits"].float()
        fruit_types = observations["fruit_types"].float()
        fruit_mask = observations["fruit_mask"].float()

        per_fruit = torch.cat([fruits, fruit_types], dim=-1)
        encoded = self.fruit_encoder(per_fruit)

        mask = fruit_mask.unsqueeze(-1)
        active_count = mask.sum(dim=1).clamp(min=1.0)
        mean_pool = (encoded * mask).sum(dim=1) / active_count

        max_input = encoded.masked_fill(mask == 0.0, float("-inf"))
        max_pool, _ = max_input.max(dim=1)
        any_active = fruit_mask.any(dim=1, keepdim=True)
        max_pool = torch.where(any_active, max_pool, torch.zeros_like(max_pool))

        batch_size = fruits.shape[0]
        current_fruit = self._spawn_one_hot(observations["current_fruit"], batch_size)
        next_fruit = self._spawn_one_hot(observations["next_fruit"], batch_size)

        return torch.cat([mean_pool, max_pool, current_fruit, next_fruit], dim=-1)

    def _spawn_one_hot(self, value: torch.Tensor, batch_size: int) -> torch.Tensor:
        if value.shape[-1:] == (self.n_spawn_types,):
            return value.float().view(batch_size, self.n_spawn_types)
        return F.one_hot(
            value.long().view(batch_size),
            num_classes=self.n_spawn_types,
        ).float()
