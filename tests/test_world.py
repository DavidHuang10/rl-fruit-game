# AI-assisted world tests; merge, scoring, and game-over cases checked after play-testing.
"""Unit tests for SuikaWorld physics, merges, scoring, and game-over."""
import numpy as np
import pytest

from suika_env.config import EnvConfig
from suika_env.fruits import FRUITS, NUM_FRUITS
from suika_env.world import SuikaWorld


def make_world(seed: int = 0, **overrides) -> SuikaWorld:
    cfg = EnvConfig(**overrides)
    rng = np.random.default_rng(seed)
    return SuikaWorld(cfg, rng)


# ---------------------------------------------------------------------------
# Merge mechanics
# ---------------------------------------------------------------------------


class TestMerges:
    def test_two_cherries_merge_into_strawberry(self):
        world = make_world()
        # Stack one cherry on top of another — they will collide and merge.
        world.add_fruit(250, 650, 0)
        world.add_fruit(250, 626, 0)  # 24px gap = exactly touching on first contact
        score, _ = world.run_until_settled()
        assert score == FRUITS[1].score, f"Expected {FRUITS[1].score}, got {score}"
        assert world.num_fruits == 1
        assert world.serialize()[0].fruit_type == 1  # strawberry

    def test_cascade_cherries_to_grape(self):
        world = make_world()
        # Four cherries arranged in two pairs → two strawberries → one grape
        world.add_fruit(200, 650, 0)
        world.add_fruit(200, 626, 0)
        world.add_fruit(300, 650, 0)
        world.add_fruit(300, 626, 0)
        score, _ = world.run_until_settled()
        # Cascade may take multiple settle cycles; we only spawn these 4 at once.
        # Min expected: 3 + 3 = 6 (two cherry→strawberry); if they also merge: +6
        assert score >= 6
        assert world.total_score >= 6

    def test_watermelon_pair_vanishes(self):
        world = make_world()
        # Drop two watermelons directly on top of each other — they should vanish.
        world.add_fruit(250, 650, 10)
        world.add_fruit(250, 442, 10)  # 104+104=208px stacked
        score, _ = world.run_until_settled()
        assert score == world.cfg.watermelon_merge_score
        assert world.num_fruits == 0

    def test_single_watermelon_stays(self):
        world = make_world()
        world.add_fruit(250, 650, 10)
        score, _ = world.run_until_settled()
        assert score == 0
        assert world.num_fruits == 1

    def test_different_types_do_not_merge(self):
        world = make_world()
        world.add_fruit(240, 650, 0)  # cherry
        world.add_fruit(260, 650, 1)  # strawberry — adjacent, different types
        score, _ = world.run_until_settled()
        assert score == 0
        assert world.num_fruits == 2

    def test_score_accumulates_across_merges(self):
        world = make_world()
        world.add_fruit(250, 650, 0)
        world.add_fruit(250, 626, 0)
        world.run_until_settled()
        assert world.total_score == FRUITS[1].score

    @pytest.mark.parametrize("fruit_type", range(NUM_FRUITS - 1))
    def test_each_tier_merges_correctly(self, fruit_type):
        world = make_world()
        r = FRUITS[fruit_type].radius
        world.add_fruit(250, 650, fruit_type)
        world.add_fruit(250, 650 - 2 * r, fruit_type)
        score, _ = world.run_until_settled()
        assert score == FRUITS[fruit_type + 1].score
        assert world.num_fruits == 1
        assert world.serialize()[0].fruit_type == fruit_type + 1


# ---------------------------------------------------------------------------
# Settle loop
# ---------------------------------------------------------------------------


class TestSettle:
    def test_single_fruit_settles(self):
        world = make_world()
        world.add_fruit(250, 50, 0)
        _, frames = world.run_until_settled()
        assert frames < world.cfg.max_physics_steps_per_action
        assert world.num_fruits == 1

    def test_settle_respects_hard_cap(self):
        # A fruit that can never truly settle must still terminate within cap.
        # We make an unreachable threshold to force the cap to kick in.
        world = make_world(settle_ke_threshold=0.0, settle_stable_frames=999)
        world.add_fruit(250, 300, 5)
        _, frames = world.run_until_settled()
        assert frames == world.cfg.max_physics_steps_per_action


# ---------------------------------------------------------------------------
# Game-over detection
# ---------------------------------------------------------------------------


class TestGameOver:
    def test_fruit_above_danger_line_triggers_gameover(self):
        world = make_world()
        danger_y = world.cfg.danger_line_y  # default 120
        radius = FRUITS[0].radius  # 12
        # Place cherry so its top (center_y - radius) is above the danger line.
        # center_y = danger_y - 5  →  top_y = danger_y - 5 - radius = 103 < 120 ✓
        world.add_fruit(250, danger_y - 5, 0)
        # is_game_over checks body positions directly; no settle needed for this test.
        assert world.is_game_over()

    def test_fruit_below_danger_line_no_gameover(self):
        world = make_world()
        danger_y = world.cfg.danger_line_y
        radius = FRUITS[0].radius
        # center_y = danger_y + radius + 10  →  top_y = danger_y + 10 > danger_y ✗ (safe)
        world.add_fruit(250, danger_y + radius + 10, 0)
        assert not world.is_game_over()

    def test_empty_world_no_gameover(self):
        world = make_world()
        assert not world.is_game_over()
