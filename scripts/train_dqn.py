# AI-assisted wrapper for DQN training; delegates to the reviewed src implementation.
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "src" / "scripts" / "train_dqn.py"), run_name="__main__")
