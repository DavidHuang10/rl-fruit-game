from dataclasses import dataclass
from typing import Tuple

NUM_FRUITS = 11
NEXT_FRUIT_POOL = list(range(5))  # cherry → persimmon are agent-spawnable


@dataclass(frozen=True)
class FruitDef:
    idx: int
    name: str
    radius: int
    score: int
    color: Tuple[int, int, int]


FRUITS: Tuple[FruitDef, ...] = (
    FruitDef(0,  "Cherry",      12,  1,  (220,  50,  50)),
    FruitDef(1,  "Strawberry",  18,  3,  (240,  80,  80)),
    FruitDef(2,  "Grape",       24,  6,  (150,  80, 200)),
    FruitDef(3,  "Dekopon",     30, 10,  (255, 160,  30)),
    FruitDef(4,  "Persimmon",   38, 15,  (255, 100,  20)),
    FruitDef(5,  "Apple",       46, 21,  (180,  30,  30)),
    FruitDef(6,  "Pear",        56, 28,  (210, 210,  80)),
    FruitDef(7,  "Peach",       66, 36,  (255, 180, 140)),
    FruitDef(8,  "Pineapple",   78, 45,  (230, 210,  50)),
    FruitDef(9,  "Melon",       90, 55,  (120, 200,  80)),
    FruitDef(10, "Watermelon", 104, 66,  ( 40, 160,  60)),
)
