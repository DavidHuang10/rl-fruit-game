from __future__ import annotations

import argparse
import pathlib
import random

import pygame

from agents import SelectsAction, build_agent
from suika_env.env import SuikaEnv
from suika_env.fruits import FRUITS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch an agent play SuikaEnv in real time.")
    p.add_argument("--agent", choices=["random", "center", "dqn", "ppo"], required=True)
    p.add_argument("--checkpoint", type=pathlib.Path, default=None)
    p.add_argument("--seed", type=int, default=None, help="Fixed RNG seed. Omit for a random seed each episode.")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. Values above 1 run extra physics ticks per frame.",
    )
    p.add_argument(
        "--pause-on-game-over",
        action="store_true",
        help="Wait for R/Q after each finished episode instead of auto-advancing.",
    )
    return p.parse_args()


def draw_aim(env: SuikaEnv, col: int) -> None:
    if env._screen is None:
        return
    cfg = env.cfg
    fruit_r = FRUITS[env._current_fruit].radius
    x_min = cfg.wall_thickness + fruit_r
    x_max = cfg.container_width - cfg.wall_thickness - fruit_r
    x = int(x_min + (col + 0.5) / cfg.n_action_bins * (x_max - x_min))
    for y in range(cfg.spawn_y, cfg.container_height, 8):
        pygame.draw.line(
            env._screen,
            (150, 150, 220),
            (x, y),
            (x, min(y + 4, cfg.container_height)),
            1,
        )


def poll_quit_or_restart() -> str | None:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_q, pygame.K_ESCAPE):
                return "quit"
            if event.key == pygame.K_r:
                return "restart"
    return None


def wait_after_game_over(clock: pygame.time.Clock, fps: int) -> str:
    while True:
        event = poll_quit_or_restart()
        if event in {"quit", "restart"}:
            return event
        clock.tick(fps)


def run_episode(
    env: SuikaEnv,
    agent: SelectsAction,
    seed: int,
    fps: int,
    speed: float,
) -> tuple[bool, float, int]:
    obs, _ = env.reset(seed=seed)
    clock = pygame.time.Clock()
    episode_return = 0.0
    episode_len = 0

    while True:
        event = poll_quit_or_restart()
        if event == "quit":
            return False, episode_return, episode_len
        if event == "restart":
            obs, _ = env.reset(seed=seed)
            episode_return = 0.0
            episode_len = 0

        action = agent.select_action(obs, eval_mode=True)
        env.drop(action)

        terminated = False
        reward = 0.0
        tick_budget = 0.0
        while True:
            event = poll_quit_or_restart()
            if event == "quit":
                return False, episode_return, episode_len
            if event == "restart":
                obs, _ = env.reset(seed=seed)
                episode_return = 0.0
                episode_len = 0
                break

            tick_budget += speed
            settled = False
            while tick_budget >= 1.0 and not settled:
                frame_score, settled, terminated = env.tick()
                reward += float(frame_score)
                tick_budget -= 1.0
            env.render()
            draw_aim(env, action)
            pygame.display.flip()
            clock.tick(fps)

            if settled:
                obs = env._get_obs()
                episode_return += reward
                episode_len += 1
                break

        if terminated:
            assert env._world is not None
            print(
                f"Game over: score={env._world.total_score}, "
                f"return={episode_return:.0f}, steps={episode_len}"
            )
            return True, episode_return, episode_len


def main() -> None:
    args = parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be greater than 0")
    env = SuikaEnv(render_mode="human")
    try:
        agent = build_agent(args.agent, env.action_space, args.checkpoint)
        for ep in range(args.episodes):
            episode_seed = (args.seed + ep) if args.seed is not None else random.randint(0, 2**31)
            finished, _, _ = run_episode(
                env,
                agent,
                episode_seed,
                args.fps,
                args.speed,
            )
            if not finished:
                break
            if args.pause_on_game_over and ep < args.episodes - 1:
                event = wait_after_game_over(env._clock or pygame.time.Clock(), args.fps)
                if event == "quit":
                    break
    finally:
        env.close()


def _main_for(agent: str) -> None:
    import sys
    sys.argv = [sys.argv[0], "--agent", agent] + sys.argv[1:]
    main()


def main_dqn() -> None:
    _main_for("dqn")


def main_ppo() -> None:
    _main_for("ppo")


def main_center() -> None:
    _main_for("center")


def main_random() -> None:
    _main_for("random")


if __name__ == "__main__":
    main()
