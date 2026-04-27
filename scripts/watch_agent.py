# AI-assisted wrapper for watching agents; used for qualitative evaluation clips.
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
runpy.run_path(str(ROOT / "src" / "scripts" / "watch_agent.py"), run_name="__main__")
