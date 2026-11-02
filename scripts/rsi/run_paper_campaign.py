"""Persistent multi-paper reproduction, registration and independently checked reuse.

Each admitted partial scope is frozen before execution. No historical success,
model statement, or completed subset can satisfy an unverified requested scope.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from paper_library import bind_completed_recipe, import_recipes, publish_campaign
from reproduction_continuation import POLICY as REVIEW_CONTINUATION
from run_harness_probe import ROOT, credential_from_toml, default_runtimes
from run_harness_reproduction import archive_implementation
from run_reproduction_goal import ReproductionGoal, read

from agent.harness import Profile
from agent.harness.host import write_json
from agent.rsi.campaigns.case_plugins import invocation
from agent.rsi.campaigns.paper_campaign import SCOPES, SUITES, admitted_campaign, campaign_cases
from agent.rsi.engine.paper_capability_library import PaperCapabilityLibrary
from agent.rsi.engine.reproduction_goal_contract import CRITERIA
from agent.rsi.engine.source_snapshot import research_implementation_paths
from agent.science_goal import GoalSpec, GoalStore, run_goal


class PaperCampaign:
    def __init__(self, store, credential):
        self.store = store
        self.protocol = read(store.root / "protocol.json")
        self.papers = admitted_campaign(self.protocol, store.read())
        self.units = {p["case"]: ReproductionGoal(store, credential, case=p["case"]) for p in self.papers}

    @staticmethod
    def view(state, case):
        return {**state, "progress": state["progress"].get("papers", {}).get(case, {})}

    def evaluate(self, state):
        papers = {case: unit.evaluate(self.view(state, case)) for case, unit in self.units.items()}
        checks = {}
        for key in CRITERIA:
            passed = all(a["criteria"][key]["passed"] for a in papers.values())
            checks[key] = {
                "passed": passed,
                "evidence": [e for a in papers.values() for e in a["criteria"][key]["evidence"]] if passed else [],
            }
        return {"criteria": checks, "papers": papers, "full_paper_reproduced": False}

    def select(self, state):
        assessments = self.evaluate(state)["papers"]
        pending = [c for c in self.units if not all(v["passed"] for v in assessments[c]["criteria"].values())]
        if not pending:
            raise ValueError("Campaign already verified")
        # Fair scheduling prevents one failing paper from starving the others.
        attempts = dict.fromkeys(self.units, 0)
        for cycle in state["cycles"]:
            case = cycle.get("case")
            if case not in attempts:
                path = self.store.root / "episodes" / f"cycle_{cycle['index']:03d}" / "paper.json"
                if path.is_file():
                    case = read(path).get("case")
            if case in attempts:
                attempts[case] += 1
        return min(pending, key=lambda c: attempts[c])

    async def cycle(self, index, profile, state, context):
        case = self.select(state)
        paper = next(p for p in self.papers if p["case"] == case)
        # Preserve the chosen paper even if the worker is interrupted before checkpoint.
        path = self.store.root / "episodes" / f"cycle_{index:03d}"
        path.mkdir(parents=True, exist_ok=False)
        write_json(path / "paper.json", paper)
        catalog = [
            {"case": c, "recipe_version": p["recipe_version"]}
            for c, p in state["progress"].get("papers", {}).items()
            if p.get("recipe_version")
        ]
        context += "\nCurrent admitted paper scope:\n" + json.dumps(paper)
        context += "\nPreviously registered campaign capabilities:\n" + json.dumps(catalog)
        context += "\nMeet this scope exactly. Report exclusions; do not claim full-paper or cross-protein performance."
        try:
            outcome = await self.units[case].cycle(index, profile, self.view(state, case), context, prepared=True)
        except Exception as exc:
            # Unknown usage retains the supervisor's full reservation. Never reset
            # a failed attempt or publish a partially created recipe.
            outcome = {
                "status": "failed",
                "phase": "attempt",
                "error_type": type(exc).__name__,
                "summary": "Paper episode failed before verified checkpoint; evidence and reserved usage retained",
                "progress": self.view(state, case)["progress"],
            }
        progress = {**state["progress"], "papers": dict(state["progress"].get("papers", {}))}
        progress["papers"][case] = outcome["progress"]
        outcome.update(case=case, paper_id=paper["paper_id"], progress=progress)
        outcome["summary"] = case + ": " + outcome["summary"]
        write_json(path / "outcome.json", outcome)
        return outcome


def prepare(args):
    papers = campaign_cases(args.case, getattr(args, "paper_case", None))
    if args.max_cycles < 2 * len(papers):
        raise ValueError("Allow at least two episodes per paper scope, for reproduction and fresh reuse")
    profile = Profile.from_dict(read(args.profile))
    for paper in papers:
        declaration = paper["allocation"]
        invocation(
            declaration["plugin"],
            declaration["parameters"],
            root=ROOT,
            output=args.output / "preflight" / paper["case"],
            controller_python=Path(default_runtimes()["python_bin"]),
        )
    goal = GoalSpec(
        args.objective,
        CRITERIA,
        max_cycles=args.max_cycles,
        model_calls=args.model_calls,
        tool_calls=args.tool_calls,
        wall_seconds=args.wall_seconds,
    )
    library, matches = None, {}
    admission = {"kind": "paper_campaign.v1", "papers": papers, "review_continuation": REVIEW_CONTINUATION}
    if getattr(args, "capability_library", None):
        library = PaperCapabilityLibrary(args.capability_library)
        library.goal(args.output)
        for paper in papers:
            item = library.find(paper, bind_completed_recipe)
            if item is not None:
                matches[paper["case"]] = item
        admission["capability_library"] = {
            "path": str(library.root),
            "imports": {case: item["entry_id"] for case, item in matches.items()},
        }
    store = GoalStore.create(
        args.output,
        goal,
        profile,
        condition=getattr(args, "condition", "science_goal"),
        admission=admission,
    )
    hashes = archive_implementation(research_implementation_paths(ROOT), store.root)
    write_json(
        store.root / "protocol.json",
        {
            "kind": "paper_campaign.v1",
            "review_continuation": REVIEW_CONTINUATION,
            "papers": papers,
            **({"capability_library": admission["capability_library"]} if library else {}),
            "implementation": hashes,
            "cycle_wall_seconds": max(p["allocation"]["job_timeout"] for p in papers) + profile.budget.wall_seconds,
            "scope": "Declared partial computational reproductions and verified reusable workflows; model invention is evaluated separately",
        },
    )
    if library:
        import_recipes(store, library, matches)
    return store


async def main(args):
    store = prepare(args) if args.action == "create" else GoalStore(args.output)
    if args.action == "publish":
        publish_campaign(store)
        state = store.read()
    elif args.action == "pause":
        state = store.pause()
    elif args.action in {"create", "status"}:
        state = store.read()
    else:
        if args.action == "resume":
            store.resume()
        profile = Profile.from_dict(store.read()["definition"]["profile"])
        key = (
            credential_from_toml(args.credential_toml, profile.model)
            if args.credential_toml
            else os.environ.get(profile.model.credential_env)
        )
        if not key:
            raise ValueError("Goal model credential is unresolved")
        campaign = PaperCampaign(store, key)
        state = await run_goal(
            store, campaign.cycle, campaign.evaluate, cycle_wall_seconds=campaign.protocol["cycle_wall_seconds"]
        )
        if state["status"] == "completed":
            publish_campaign(store)
    print(
        json.dumps(
            {
                "status": state["status"],
                "episodes": len(state["cycles"]),
                "usage": state["usage"],
                "criteria": state["assessment"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "run", "pause", "resume", "status", "publish"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument(
        "--capability-library",
        type=Path,
        help="Owner-scoped library; source and destination goals must lie below its parent",
    )
    parser.add_argument("--objective")
    parser.add_argument("--case", choices=sorted(SUITES), default="cpu_papers")
    parser.add_argument(
        "--paper-case",
        action="append",
        choices=sorted(SCOPES),
        help="Explicit scopes; repeat to compose a custom campaign",
    )
    parser.add_argument("--max-cycles", type=int, default=8)
    parser.add_argument("--model-calls", type=int, default=192)
    parser.add_argument("--tool-calls", type=int, default=96)
    parser.add_argument("--wall-seconds", type=float, default=21600)
    parser.add_argument("--credential-toml", type=Path)
    asyncio.run(main(parser.parse_args()))
