from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pymunk

from .config import EnvConfig
from .fruits import FRUITS, NUM_FRUITS


@dataclass
class FruitState:
    x: float
    y: float
    vx: float
    vy: float
    fruit_type: int


class SuikaWorld:
    def __init__(self, config: EnvConfig, rng: np.random.Generator) -> None:
        self.cfg = config
        self.rng = rng
        self._next_id: int = 0
        self._pending_merges: List[Tuple[pymunk.Body, pymunk.Body]] = []
        self._alive_ids: Set[int] = set()
        self._body_map: Dict[int, pymunk.Body] = {}

        self.space = pymunk.Space()
        self.space.gravity = (0, config.gravity)
        self.space.damping = 0.25

        self._add_walls()
        self.space.on_collision(None, None, begin=self._on_collision_begin)

        self.total_score: int = 0
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _add_walls(self) -> None:
        cfg = self.cfg
        w, h = cfg.container_width, cfg.container_height
        t = cfg.wall_thickness
        # Bottom segment center sits at h+t so its top collision surface lands exactly at h,
        # which lines up with the visual wall top drawn by the renderer.
        floor = pymunk.Segment(self.space.static_body, (0, h + t), (w, h + t), t)
        floor.friction = 3.0
        floor.elasticity = 0.1
        sides = [
            pymunk.Segment(self.space.static_body, (0, 0), (0, h + t), t),   # left
            pymunk.Segment(self.space.static_body, (w, 0), (w, h + t), t),   # right
        ]
        for wall in sides:
            wall.friction = 0.5
            wall.elasticity = 0.1
        self.space.add(floor, *sides)

    def _on_collision_begin(
        self, arbiter: pymunk.Arbiter, space: pymunk.Space, data: object
    ) -> None:
        b_a, b_b = arbiter.bodies
        # Ignore collisions involving static/kinematic bodies (walls)
        if b_a.body_type != pymunk.Body.DYNAMIC or b_b.body_type != pymunk.Body.DYNAMIC:
            return
        if not (hasattr(b_a, "suika_id") and hasattr(b_b, "suika_id")):
            return
        if b_a.suika_type != b_b.suika_type:
            return
        # Same type → schedule merge; deduplicate by sorted id pair.
        pair = (b_a, b_b) if b_a.suika_id < b_b.suika_id else (b_b, b_a)
        ids_in_queue = {(a.suika_id, b.suika_id) for a, b in self._pending_merges}
        if (pair[0].suika_id, pair[1].suika_id) not in ids_in_queue:
            self._pending_merges.append(pair)

    # ------------------------------------------------------------------
    # Fruit management
    # ------------------------------------------------------------------

    def add_fruit(
        self,
        x: float,
        y: float,
        fruit_type: int,
        vx: float = 0.0,
        vy: float = 0.0,
    ) -> pymunk.Body:
        fdef = FRUITS[fruit_type]
        radius = fdef.radius
        mass = self.cfg.fruit_density * math.pi * radius**2

        body = pymunk.Body(mass=mass, moment=pymunk.moment_for_circle(mass, 0, radius))
        body.position = (x, y)
        body.velocity = (vx, vy)

        shape = pymunk.Circle(body, radius)
        shape.friction = self.cfg.fruit_friction
        shape.elasticity = self.cfg.fruit_elasticity

        body.suika_id = self._next_id
        body.suika_type = fruit_type
        self._next_id += 1

        self.space.add(body, shape)
        self._alive_ids.add(body.suika_id)
        self._body_map[body.suika_id] = body
        return body

    def _remove_body(self, body: pymunk.Body) -> None:
        sid = body.suika_id
        if sid not in self._alive_ids:
            return
        self._alive_ids.discard(sid)
        self._body_map.pop(sid, None)
        self.space.remove(body, *body.shapes)

    # ------------------------------------------------------------------
    # Merge processing
    # ------------------------------------------------------------------

    def _process_merges(self) -> int:
        if not self._pending_merges:
            return 0

        score_gained = 0
        seen: Set[Tuple[int, int]] = set()
        to_process = self._pending_merges[:]
        self._pending_merges.clear()

        for b_a, b_b in to_process:
            key = (b_a.suika_id, b_b.suika_id)
            if key in seen:
                continue
            seen.add(key)
            if b_a.suika_id not in self._alive_ids or b_b.suika_id not in self._alive_ids:
                continue

            fruit_type = b_a.suika_type
            mid_x = (b_a.position.x + b_b.position.x) / 2
            mid_y = (b_a.position.y + b_b.position.y) / 2
            avg_vx = (b_a.velocity.x + b_b.velocity.x) / 2
            avg_vy = (b_a.velocity.y + b_b.velocity.y) / 2

            self._remove_body(b_a)
            self._remove_body(b_b)

            if fruit_type == NUM_FRUITS - 1:
                # Watermelon + watermelon → both vanish, bonus score
                score_gained += self.cfg.watermelon_merge_score
            else:
                new_type = fruit_type + 1
                self.add_fruit(mid_x, mid_y, new_type, avg_vx, avg_vy)
                score_gained += FRUITS[new_type].score

        self.total_score += score_gained
        return score_gained

    # ------------------------------------------------------------------
    # Physics stepping
    # ------------------------------------------------------------------

    def step_physics(self) -> int:
        sub_dt = self.cfg.physics_dt / self.cfg.physics_substeps
        for _ in range(self.cfg.physics_substeps):
            self.space.step(sub_dt)
        self._step_count += 1
        return self._process_merges()

    def run_until_settled(self) -> Tuple[int, int]:
        score_gained = 0
        quiet_frames = 0
        frames = 0

        for _ in range(self.cfg.max_physics_steps_per_action):
            frame_score = self.step_physics()
            frames += 1
            score_gained += frame_score

            if frame_score > 0:
                quiet_frames = 0
                continue

            ke = self._total_kinetic_energy()
            if ke < self.cfg.settle_ke_threshold:
                quiet_frames += 1
            else:
                quiet_frames = 0

            if quiet_frames >= self.cfg.settle_stable_frames:
                break

        return score_gained, frames

    def _total_kinetic_energy(self) -> float:
        total = 0.0
        for sid in self._alive_ids:
            body = self._body_map[sid]
            v = body.velocity
            total += 0.5 * body.mass * (v.x * v.x + v.y * v.y)
        return total

    def zero_velocities(self) -> None:
        """Zero linear and angular velocity on every fruit. Called after settling."""
        for sid in self._alive_ids:
            body = self._body_map[sid]
            body.velocity = (0.0, 0.0)
            body.angular_velocity = 0.0

    # ------------------------------------------------------------------
    # Game-over check (call only after settling)
    # ------------------------------------------------------------------

    def is_game_over(self) -> bool:
        for sid in self._alive_ids:
            body = self._body_map[sid]
            shapes = list(body.shapes)
            if not shapes:
                continue
            radius = shapes[0].radius
            top_y = body.position.y - radius
            if top_y < self.cfg.danger_line_y:
                return True
        return False

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def serialize(self) -> List[FruitState]:
        return [
            FruitState(
                x=self._body_map[sid].position.x,
                y=self._body_map[sid].position.y,
                vx=self._body_map[sid].velocity.x,
                vy=self._body_map[sid].velocity.y,
                fruit_type=self._body_map[sid].suika_type,
            )
            for sid in sorted(self._alive_ids)
        ]

    @property
    def num_fruits(self) -> int:
        return len(self._alive_ids)
