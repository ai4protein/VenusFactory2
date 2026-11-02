"""Private recipient acceptance for an immutable generated source import."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi.paper_step_transfer import accept_import
from agent.science_goal import GoalStore

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--goal", type=Path, required=True)
p.add_argument("--folder", type=Path, required=True)
a = p.parse_args()
accept_import(GoalStore(a.goal), a.folder)
