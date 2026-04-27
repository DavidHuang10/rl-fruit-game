# AI-assisted wrapper for PPO training; delegates to the reviewed SB3 training script.
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "src" / "scripts" / "train_ppo_sb3.py"), run_name="__main__")
