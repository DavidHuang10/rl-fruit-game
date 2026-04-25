"""
Keyboard-controlled human play mode for SuikaEnv.

Controls:
  LEFT / RIGHT arrow : move drop column (hold to scroll)
  SPACE or RETURN    : drop fruit / skip animation
  R                  : restart
  Q or ESC           : quit
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pygame
from suika_env.env import SuikaEnv
from suika_env.config import EnvConfig


def _settle_one_tick(env: SuikaEnv) -> tuple[bool, bool]:
    """Advance one physics tick. Returns (settled, terminated)."""
    frame_score, settled, terminated = env.tick()
    if frame_score > 0:
        assert env._world is not None
        print(f"  Merge! +{int(frame_score)}  (total {env._world.total_score})")
    return settled, terminated


def _settle_to_end(env: SuikaEnv) -> bool:
    """Run physics to completion. Returns terminated."""
    while True:
        settled, terminated = _settle_one_tick(env)
        if settled:
            return terminated


def _on_settled(env: SuikaEnv, terminated: bool) -> str:
    if terminated:
        assert env._world is not None
        print(f"\nGame Over! Final score: {env._world.total_score}")
        print("Press R to restart or Q to quit.")
        return "game_over"
    return "waiting"


def main() -> None:
    cfg = EnvConfig()
    env = SuikaEnv(config=cfg, render_mode="human")
    env.reset(seed=None)
    pygame.key.set_repeat(200, 50)

    n_bins = cfg.n_action_bins
    col = n_bins // 2
    clock = pygame.time.Clock()
    state = "waiting"
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    col = n_bins // 2
                    state = "waiting"
                    print("Restarted.")
                elif state == "dropping" and event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    state = _on_settled(env, _settle_to_end(env))
                elif state == "waiting":
                    if event.key == pygame.K_LEFT:
                        col = max(0, col - 1)
                    elif event.key == pygame.K_RIGHT:
                        col = min(n_bins - 1, col + 1)
                    elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        env.drop(col)
                        state = "dropping"

        if state == "dropping":
            settled, terminated = _settle_one_tick(env)
            if settled:
                state = _on_settled(env, terminated)

        env.render()
        if state == "waiting":
            _draw_aim(env, col)
        pygame.display.flip()
        clock.tick(60)

    env.close()


def _draw_aim(env: SuikaEnv, col: int) -> None:
    if env._screen is None:
        return
    cfg = env.cfg
    from suika_env.fruits import FRUITS
    fruit_r = FRUITS[env._current_fruit].radius
    x_min = cfg.wall_thickness + fruit_r
    x_max = cfg.container_width - cfg.wall_thickness - fruit_r
    x = int(x_min + (col + 0.5) / cfg.n_action_bins * (x_max - x_min))
    for y in range(cfg.spawn_y, cfg.container_height, 8):
        pygame.draw.line(env._screen, (150, 150, 220), (x, y), (x, min(y + 4, cfg.container_height)), 1)


if __name__ == "__main__":
    main()
