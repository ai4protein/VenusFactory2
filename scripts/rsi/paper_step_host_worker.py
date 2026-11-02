"""Private, cancellable host verification subprocess; never an agent tool."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from agent.rsi.paper_step import PaperStepLifecycle
from agent.rsi.paper_step_sessions import accept_development, accept_reuse, read

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--goal", type=Path, required=True)
parser.add_argument("--folder", type=Path, required=True)
parser.add_argument("--phase", choices=["develop", "reuse"], required=True)
parser.add_argument("--development-session")
args = parser.parse_args()
args.folder.resolve().relative_to(args.goal.resolve())
definition = read(args.goal / "step_definition.json")
context = read(args.goal / "protocol.json").get("scientist_tool_context")
if context:
    from agent.rsi.research_tool_lifecycle import ResearchToolLifecycle
    from agent.rsi.scientist_tool_context import verify_context

    verify_context(args.folder, context, definition)
    lc = ResearchToolLifecycle(args.goal / "host_private", definition, context["contract"])
else:
    lc = PaperStepLifecycle(args.goal / "host_private", definition)
if args.phase == "develop":
    accept_development(lc, args.folder)
else:
    if not args.development_session:
        parser.error("reuse requires development session identity")
    accept_reuse(lc, args.folder, development_session=args.development_session)
