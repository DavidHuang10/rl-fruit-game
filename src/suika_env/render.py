# AI-assisted renderer, claude code + Codex.
from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from .config import EnvConfig
from .fruits import FRUITS

if TYPE_CHECKING:
    from .world import SuikaWorld

# -----------------------------------------------------------------------
# Colours
# -----------------------------------------------------------------------
BG_COLOR = (255, 245, 220)
WALL_COLOR = (180, 140, 90)
DANGER_LINE_COLOR = (200, 50, 50)
HUD_BG_COLOR = (240, 220, 180)
TEXT_COLOR = (60, 40, 20)
OUTLINE_COLOR = (0, 0, 0)


def render_frame(
    world: "SuikaWorld",
    surface: pygame.Surface,
    cfg: EnvConfig,
    current_fruit: int,
    next_fruit: int,
    show_danger_line: bool = True,
) -> pygame.Surface:
    """Draw the current world state onto *surface* and return it."""
    surface.fill(BG_COLOR)

    w, h = cfg.container_width, cfg.container_height
    t = cfg.wall_thickness

    # Container walls
    pygame.draw.rect(surface, WALL_COLOR, (0, 0, t, h + t))  # left
    pygame.draw.rect(surface, WALL_COLOR, (w - t, 0, t, h + t))  # right
    pygame.draw.rect(surface, WALL_COLOR, (0, h, w, t + 2))  # bottom

    # Danger line
    if show_danger_line:
        dy = int(cfg.danger_line_y)
        for x in range(t, w - t, 12):
            pygame.draw.line(
                surface, DANGER_LINE_COLOR, (x, dy), (min(x + 6, w - t), dy), 2
            )

    # Fruits
    for fs in world.serialize():
        fdef = FRUITS[fs.fruit_type]
        cx = int(fs.x)
        cy = int(fs.y)
        r = fdef.radius
        pygame.draw.circle(surface, fdef.color, (cx, cy), r)
        pygame.draw.circle(surface, OUTLINE_COLOR, (cx, cy), r, 1)

    # HUD panel (right side or top-right strip)
    _draw_hud(surface, cfg, current_fruit, next_fruit, world.total_score)

    return surface


def _draw_hud(
    surface: pygame.Surface,
    cfg: EnvConfig,
    current_fruit: int,
    next_fruit: int,
    score: int,
) -> None:
    font = _get_font(18)
    small_font = _get_font(14)

    pad = 8
    panel_x = cfg.container_width + 4

    # Score
    score_surf = font.render(f"Score: {score}", True, TEXT_COLOR)
    surface.blit(score_surf, (panel_x, pad))

    # Current fruit preview
    label = small_font.render("Now:", True, TEXT_COLOR)
    surface.blit(label, (panel_x, pad + 30))
    _draw_fruit_preview(surface, panel_x + 60, pad + 45, current_fruit)

    # Next fruit preview
    label2 = small_font.render("Next:", True, TEXT_COLOR)
    surface.blit(label2, (panel_x, pad + 80))
    _draw_fruit_preview(surface, panel_x + 60, pad + 95, next_fruit)


def _draw_fruit_preview(
    surface: pygame.Surface, cx: int, cy: int, fruit_type: int
) -> None:
    fdef = FRUITS[fruit_type]
    r = min(fdef.radius, 22)  # cap preview size so it fits in HUD
    pygame.draw.circle(surface, fdef.color, (cx, cy), r)
    pygame.draw.circle(surface, OUTLINE_COLOR, (cx, cy), r, 1)
    font = _get_font(11)
    label = font.render(fdef.name[:4], True, TEXT_COLOR)
    surface.blit(label, (cx - label.get_width() // 2, cy + r + 2))


_font_cache: dict[int, pygame.font.Font] = {}


def _get_font(size: int) -> pygame.font.Font:
    if size not in _font_cache:
        pygame.font.init()
        _font_cache[size] = pygame.font.SysFont("sans", size)
    return _font_cache[size]
