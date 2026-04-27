# AI-assisted configuration; physics and game-over constants tuned through my play-testing.
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional

from .constants import ACTION_COUNT, MAX_FRUITS


# NOTE: I had to change these parameters to match how I thought the original game appeared, editing gravity, friction, collision elasticity, etc.
@dataclass(frozen=True)
class EnvConfig:
    # Container geometry (px)
    container_width: int = 500
    container_height: int = 700
    wall_thickness: int = 10

    # Drop mechanics
    spawn_y: int = 50  # y where new fruit appears (below top edge)
    danger_line_y: int = 120  # fruits resting above this → game over

    # Action space
    n_action_bins: int = ACTION_COUNT

    # Observation
    max_fruits: int = MAX_FRUITS
    observation_mode: Literal["state", "pixels"] = "state"

    # Physics
    gravity: float = 900.0
    physics_dt: float = 1 / 60
    physics_substeps: int = 4  # effective 240 Hz
    fruit_friction: float = 0.4
    fruit_elasticity: float = 0.3
    fruit_density: float = 1.0

    # Settle detection
    settle_ke_threshold: float = 50.0
    settle_stable_frames: int = 8
    max_physics_steps_per_action: int = 300  # safety cap (~5 s at 60 fps)

    # Scoring
    watermelon_merge_score: int = 66

    # Reproducibility
    seed: Optional[int] = None
