"""Discover or execute a pinned protein-study reproduction plugin."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from agent.rsi.campaigns.case_plugins import invocation, list_case_plugins

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["list", "run"])
    p.add_argument("--plugin")
    p.add_argument("--parameters", default="{}")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.action == "list":
        print(json.dumps(list_case_plugins(), indent=2))
    else:
        if not args.plugin or args.output is None:
            p.error("run requires --plugin and --output")
        argv = invocation(
            args.plugin,
            json.loads(args.parameters),
            root=ROOT,
            output=args.output,
            controller_python=Path(sys.executable),
        )
        env = {
            "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
            "HOME": str(ROOT / "research_runs/rsi/case_plugin_home"),
            "PYTHONPATH": str(ROOT / "src"),
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "0",
        }
        Path(env["HOME"]).mkdir(exist_ok=True)
        raise SystemExit(subprocess.run(argv, cwd=ROOT, env=env).returncode)
