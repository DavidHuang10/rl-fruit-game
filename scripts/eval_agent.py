# AI-assisted wrapper for evaluation; preserves a simple top-level command path.
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "src" / "scripts" / "eval_agent.py"), run_name="__main__")
