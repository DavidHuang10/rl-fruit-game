# AI-assisted Gymnasium environment; observation/action/reward design directed by me.
# NOTE: I had to debug the observation collection and design how the vector representation of the fruits worked.
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from .config import EnvConfig
from .fruits import FRUITS, NUM_FRUITS, NEXT_FRUIT_POOL
from .render import render_frame
from .world import SuikaWorld

# Suppress pygame welcome banner when not rendering to a screen
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

_POOL_SIZE = len(NEXT_FRUIT_POOL)
# Velocity normalisation scale (px/s) — keeps obs values near [-1, 1]
_VEL_SCALE = 800.0
# HUD width (px) to the right of the container
_HUD_WIDTH = 150


class SuikaEnv(gym.Env):
    """Custom Gymnasium wrapper for the Suika-style physics game.

    NOTE: I designed the action bins, structured observations, and
    merge-score reward to match the original game, triangular growth instead of exponential.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.cfg = config or EnvConfig()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(self.cfg.n_action_bins)
        self.observation_space = self._build_observation_space()

        # Pygame state (created lazily on first render or when render_mode="human")
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._offscreen: Optional[pygame.Surface] = None

        # Episode state (initialised by reset)
        self._world: Optional[SuikaWorld] = None
        self._rng: Optional[np.random.Generator] = None
        self._current_fruit: int = 0
        self._next_fruit: int = 0
        self._total_score: int = 0
        self._step_num: int = 0

        # Per-drop settle tracking (used by drop()/tick())
        self._settling: bool = False
        self._settle_quiet_frames: int = 0
        self._settle_frames: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        seed_val = seed if seed is not None else self.cfg.seed
        self._rng = np.random.default_rng(seed_val)

        self._world = SuikaWorld(self.cfg, self._rng)
        self._total_score = 0
        self._step_num = 0
        self._current_fruit = int(self._rng.integers(0, _POOL_SIZE))
        self._next_fruit = int(self._rng.integers(0, _POOL_SIZE))

        if self.render_mode == "human":
            self._ensure_pygame()

        obs = self._get_obs()
        info: Dict[str, Any] = {"score": 0}
        return obs, info

    def drop(self, action: int) -> None:
        """Drop the current fruit. Call tick() each frame until it returns settled=True."""
        assert (
            self._world is not None and self._rng is not None
        ), "Call reset() before drop()"
        assert self.action_space.contains(action), f"Invalid action {action}"
        cfg = self.cfg
        fruit_radius = FRUITS[self._current_fruit].radius
        x_min = cfg.wall_thickness + fruit_radius
        x_max = cfg.container_width - cfg.wall_thickness - fruit_radius
        x = x_min + (action + 0.5) / cfg.n_action_bins * (x_max - x_min)
        self._world.add_fruit(x, cfg.spawn_y, self._current_fruit, vy=50.0)
        self._settling = True
        self._settle_quiet_frames = 0
        self._settle_frames = 0

    def tick(self) -> Tuple[float, bool, bool]:
        """Step one physics frame. Returns (frame_score, settled, terminated).

        When settled=True the turn is finalised: current/next fruit are advanced.
        Call drop() before the first tick() of each turn.
        """
        assert self._world is not None and self._rng is not None
        assert self._settling, "Call drop() before tick()"

        frame_score = self._world.step_physics()
        self._settle_frames += 1

        if frame_score > 0:
            self._settle_quiet_frames = 0
        else:
            if self._world.is_quiet():
                self._settle_quiet_frames += 1
            else:
                self._settle_quiet_frames = 0

        settled = (
            self._settle_quiet_frames >= self.cfg.settle_stable_frames
            or self._settle_frames >= self.cfg.max_physics_steps_per_action
        )

        if not settled:
            return frame_score, False, False

        self._settling = False
        self._world.zero_velocities()
        self._total_score = self._world.total_score
        self._current_fruit = self._next_fruit
        self._next_fruit = int(self._rng.integers(0, _POOL_SIZE))
        self._step_num += 1
        terminated = self._world.is_game_over()
        return frame_score, True, terminated

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        assert (
            self._world is not None and self._rng is not None
        ), "Call reset() before step()"

        self.drop(action)
        total_score = 0.0
        frames = 0
        terminated = False
        while True:
            frame_score, settled, terminated = self.tick()
            total_score += frame_score
            frames += 1
            if settled:
                break

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "score": self._total_score,
            "physics_frames": frames,
            "num_fruits": self._world.num_fruits,
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(total_score), terminated, False, info

    def render(self) -> Optional[np.ndarray]:
        assert self._world is not None, "Call reset() before render()"
        self._ensure_pygame()
        assert self._offscreen is not None
        assert self._clock is not None
        target = self._screen if self.render_mode == "human" else self._offscreen
        assert target is not None
        render_frame(
            self._world,
            target,
            self.cfg,
            self._current_fruit,
            self._next_fruit,
        )
        if self.render_mode == "human":
            pygame.display.flip()
            return None
        else:
            return np.transpose(
                np.array(pygame.surfarray.array3d(target)), axes=(1, 0, 2)
            )

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._clock = None
            self._offscreen = None

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    # NOTE: I had to debug the observation collection and design how the vector representation of the fruits worked.
    def _build_observation_space(self) -> spaces.Dict:
        cfg = self.cfg
        if cfg.observation_mode == "pixels":
            raise NotImplementedError(
                "Pixel observation mode is not yet implemented. "
                "Use observation_mode='state' (default)."
            )
        return spaces.Dict(
            {
                "fruits": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(cfg.max_fruits, 4),
                    dtype=np.float32,
                ),
                "fruit_types": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(cfg.max_fruits, NUM_FRUITS),
                    dtype=np.float32,
                ),
                "fruit_mask": spaces.MultiBinary(cfg.max_fruits),
                "current_fruit": spaces.Discrete(_POOL_SIZE),
                "next_fruit": spaces.Discrete(_POOL_SIZE),
            }
        )

    def _get_obs(self) -> Dict[str, Any]:
        cfg = self.cfg
        fruits_arr = np.zeros((cfg.max_fruits, 4), dtype=np.float32)
        types_arr = np.zeros((cfg.max_fruits, NUM_FRUITS), dtype=np.float32)
        mask_arr = np.zeros(cfg.max_fruits, dtype=np.int8)

        assert self._world is not None
        for i, fs in enumerate(self._world.serialize()):
            if i >= cfg.max_fruits:
                break
            fruits_arr[i, 0] = fs.x / cfg.container_width
            fruits_arr[i, 1] = fs.y / cfg.container_height
            fruits_arr[i, 2] = fs.vx / _VEL_SCALE
            fruits_arr[i, 3] = fs.vy / _VEL_SCALE
            types_arr[i, fs.fruit_type] = 1.0
            mask_arr[i] = 1

        return {
            "fruits": fruits_arr,
            "fruit_types": types_arr,
            "fruit_mask": mask_arr,
            "current_fruit": self._current_fruit,
            "next_fruit": self._next_fruit,
        }

    # ------------------------------------------------------------------
    # Pygame helpers
    # ------------------------------------------------------------------

    def _ensure_pygame(self) -> None:
        if self._screen is not None:
            return
        pygame.init()
        win_w = self.cfg.container_width + _HUD_WIDTH
        win_h = self.cfg.container_height + self.cfg.wall_thickness + 10
        if self.render_mode == "human":
            self._screen = pygame.display.set_mode((win_w, win_h))
            pygame.display.set_caption("Suika RL")
        else:
            # Offscreen surface for rgb_array
            self._screen = pygame.Surface((win_w, win_h))
        self._offscreen = pygame.Surface((win_w, win_h))
        self._clock = pygame.time.Clock()
