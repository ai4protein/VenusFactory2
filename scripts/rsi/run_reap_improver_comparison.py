"""Predeclare and execute a fixed/revised-improver offspring comparison.

Planning and materialization are offline. Execution is explicit. One policy
revision is shared across six tasks; it is not six independent RSI lineages.
All 18 offspring goals terminate before any higher-order future scoring.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import math
from pathlib import Path
from types import SimpleNamespace

import run_reap_goal_comparison as goals
from run_harness_probe import ROOT, default_runtimes
from run_improver_revision import SYSTEM as REVISION_SYSTEM
from run_improver_revision import TOOLS as REVISION_TOOLS
from run_improver_revision import revision_prompt
from run_reap_improver_goal import prepare as prepare_policy_goal
from run_reap_science_goal import read

from agent.harness import Profile, builtin_registry
from agent.harness.contracts import digest, file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.harness.host import write_json
from agent.rsi.improver_policy import candidate, identity, validate_candidate, validate_packet
from agent.rsi.reap_model_case import prepare_reap_model_case
from agent.rsi.versions import VersionStore
from agent.science_goal import GoalStore

CONDITIONS = ("fixed_policy", "revised_policy", "fixed_policy_total_budget")
SCOPE = (
    "Retrospective mechanism pilot on two previously inspected REAP proteins and three overlapping "
    "single-mutant partitions with shared higher-order evaluation cohorts. One predeclared policy "
    "revision is reused on six tasks; tasks are not independent policy-evolution lineages. "
    "No automatic policy promotion, confirmatory generalization or multigeneration RSI claim."
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def schedule(cases, seeds, revision_budget):
    require(tuple(cases) == goals.CASES and sorted(seeds) == [7, 8, 9]
            and all(type(x) is int for x in seeds), "Both REAP cases and seeds 7,8,9 required")
    for key in ("model_calls", "tool_calls"):
        require(type(revision_budget[key]) is int and revision_budget[key] > 0, "Invalid revision call cap")
    require(type(revision_budget["wall_seconds"]) in (int, float)
            and math.isfinite(revision_budget["wall_seconds"]) and revision_budget["wall_seconds"] > 0,
            "Invalid revision time cap")
    rows, block = [], 0
    for repeat, seed in enumerate(seeds):
        order = cases if repeat % 2 == 0 else list(reversed(cases))
        for case in order:
            extra = {key: revision_budget[key] // 6 + int(block < revision_budget[key] % 6)
                     for key in ("model_calls", "tool_calls")}
            extra["wall_seconds"] = revision_budget["wall_seconds"] / 6
            offset = block % 3
            for condition in CONDITIONS[offset:] + CONDITIONS[:offset]:
                budget = dict(goals.BUDGET)
                if condition == "fixed_policy_total_budget":
                    for key, value in extra.items():
                        budget[key] += value
                rows.append({"id": f"{case}_seed{seed:02d}_{condition}", "case": case,
                             "seed": seed, "repeat": repeat, "condition": condition,
                             "goal_budget": budget})
            block += 1
    return rows


def revision_context(root):
    root = Path(root).resolve()
    protocol = read(root / "protocol.json")
    versions = VersionStore(root / "bundles")
    versions.verify(protocol["bundle_id"])
    packet = validate_packet(read(root / "bundles/versions" / protocol["bundle_id"] / "revision_context.json"))
    require(packet["parent_id"] == protocol["parent_id"]
            and packet["history_sha256"] == protocol["history_sha256"], "Revision context binding changed")
    require(protocol["prompt_sha256"] == digest(revision_prompt(packet))
            and protocol["instructions_sha256"] == digest(REVISION_SYSTEM), "Revision instruction binding changed")
    for name, sha in protocol["implementation"].items():
        require(file_hash(root / "implementation" / name) == sha, "Revision implementation copy changed")
    for ref in protocol["history_references"]:
        for field, name in [("goal_sha256", "goal.json"), ("protocol_sha256", "protocol.json")]:
            require(file_hash(Path(ref["path"]) / name) == ref[field], "Revision source history changed")
    return protocol, packet



def token_usage(paths):
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "responses_with_usage": 0,
              "responses": 0, "requests": 0, "complete": True}
    for path in paths:
        if not path.exists():
            totals["complete"] = False
            continue
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
            if row.get("event") == "request":
                totals["requests"] += 1
            if row.get("event") != "response":
                continue
            totals["responses"] += 1
            usage = row.get("usage") or {}
            if all(type(usage.get(k)) is int and usage[k] >= 0 for k in ["prompt_tokens", "completion_tokens"]):
                totals["responses_with_usage"] += 1
                for k in ["prompt_tokens", "completion_tokens"]:
                    totals[k] += usage[k]
            else:
                totals["complete"] = False
    totals["complete"] = totals["complete"] and totals["responses"] == totals["requests"]
    totals["interpretation"] = "Provider-reported tokens summed where available; missing usage is unknown, not zero cost."
    return totals


def revision_evidence(root):
    root = Path(root).resolve()
    protocol, packet = revision_context(root)
    report = read(root / "report.json")
    require(report["status"] == "candidate_submitted", "No submitted policy; keep failed or pending revision visible")
    require(report["protocol_sha256"] == digest(protocol) and report["promoted"] is False
            and report["recursive_effectiveness_verified"] is False, "Revision status or protocol changed")
    artifact = root / "workspace/improver_candidate.json"
    require(artifact.is_file() and not artifact.is_symlink(), "Missing plain candidate artifact")
    proposed = validate_candidate(read(artifact), packet)
    profile = Profile.from_dict(protocol["profile"])
    requests = audit_requests(root / "execution/model_requests.jsonl")
    trace = audit_trace(root / "execution/events.jsonl", REVISION_TOOLS, profile.budget.tool_calls)
    run = report["run"]
    require(requests["passed"] and trace["passed"] and run["status"] == "completed"
            and report["scientific_audit"]["passed"] and run["audit"]["passed"], "Revision execution audit failed")
    require(requests["model_calls"] <= profile.budget.model_calls
            and requests == run["audit"]["requests"] and trace == run["audit"]["events"],
            "Revision physical accounting differs")
    require(report["candidate_sha256"] == identity(proposed)
            and report["policy_id"] == proposed["policy_id"]
            and report["parent_id"] == proposed["parent_id"], "Candidate report differs")
    events = [json.loads(line) for line in (root / "execution/events.jsonl").read_text().splitlines() if line.strip()]
    calls = [e for e in events if e.get("kind") == "tool_call_start"
             and e.get("tool_name", "").split("__")[-1] == "submit_improver_policy"]
    require(len(calls) == 1, "Exactly one recorded revision submission required")
    require(candidate(json.loads(calls[0]["args"]["policy_json"]), calls[0]["args"]["rationale"], packet) == proposed,
            "Candidate differs from recorded model tool call")
    elapsed = run["host_wall_seconds"]
    require(type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0, "Invalid revision elapsed time")
    names = ["protocol.json", "report.json", "workspace/improver_candidate.json",
             "execution/model_requests.jsonl", "execution/events.jsonl"]
    receipt = {"path": str(root), "files": {n: file_hash(root / n) for n in names},
               "policy_id": proposed["policy_id"], "parent_id": proposed["parent_id"],
               "history_sha256": proposed["history_sha256"],
               "observed_model_calls": requests["model_calls"], "observed_tool_calls": trace["tool_calls"],
               "host_wall_seconds": elapsed, "charged_once_per_campaign": True,
               "token_usage": token_usage([root / "execution/model_requests.jsonl"])}
    return proposed, receipt


def implementation():
    result = goals.implementation()
    for path in [Path(__file__), ROOT / "scripts/rsi/run_reap_improver_goal.py",
                 ROOT / "scripts/rsi/run_improver_revision.py"]:
        result[str(path.relative_to(ROOT))] = file_hash(path)
    return result


def copy_implementation(root, hashes):
    for name in hashes:
        dst = root / "implementation" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / name).read_bytes())


def plan(args):
    with (args.revision / ".revision.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _plan(args)


def _plan(args):
    revision = args.revision.resolve()
    rp, packet = revision_context(revision)
    require(read(revision / "report.json")["status"] == "prepared",
            "Predeclare comparison before running the policy revision")
    profile = Profile.from_dict(read(args.profile))
    require(profile.mode == "science_expert", "Matched offspring study uses the Scientist goal in every arm")
    parent_profile = Profile.from_dict(rp["profile"])
    require(profile.harness == parent_profile.harness and all(
        getattr(profile.model, k) == getattr(parent_profile.model, k)
        for k in ["provider", "model", "base_url", "protocol"]), "Keep the revision and offspring harness/backbone fixed")
    require(builtin_registry().preflight(profile, default_runtimes())["ready"], "Offspring profile unavailable")
    cap = {k: getattr(parent_profile.budget, k) for k in ["model_calls", "tool_calls", "wall_seconds"]}
    rows = schedule(list(goals.CASES), args.seeds, cap)
    hashes = implementation()
    sources = {k: str(getattr(args, k).resolve()) for k in ["admission", "features", "baseline"]}
    data = {}
    for row in rows:
        key = row["case"] + ":" + str(row["seed"])
        if key not in data:
            scope, train, val, future, source, split = prepare_reap_model_case(
                sources["admission"], sources["features"], sources["baseline"], row["case"], row["seed"])
            data[key] = {"scope": scope, "source": source, "split": split,
                         "data_hashes": {n: digest(v) for n, v in [("training", train), ("validation", val), ("future", future)]}}
    args.output.mkdir(parents=True, exist_ok=False)
    design = {"schema": 1, "schedule": rows, "profiles": {c: profile.public() for c in CONDITIONS},
              "goal_condition": "science_goal", "objective": goals.OBJECTIVE, "revision": str(revision),
              "revision_protocol_sha256": file_hash(revision / "protocol.json"), "revision_budget": cap,
              "parent_id": packet["parent_id"], "history_sha256": packet["history_sha256"],
              "sources": sources, "partitions": data, "start_after": None, "runtime_concurrency": 1,
              "primary": "paired change in complete objective attainment and development-selected higher-order ranking utility",
              "policy_effect": "revised_policy minus fixed_policy: same per-task downstream caps, tools, data and shared history",
              "budget_control": "fixed_policy_total_budget receives exactly one revision's maximum call/tool/time allowance distributed deterministically across six goals; downstream caps plus the revision allocation match at batch level, not actual realized compute",
              "cost_accounting": "Charge the one revision once to the revised batch; separately report actual physical calls, elapsed time and provider-reported tokens, with missing usage marked unknown. Matched call/tool/time caps do not imply matched tokens, FLOPs or realized compute. Shared revision is not six independent lineages.",
              "history_control": "Every offspring arm sees the identical development-history projection used for revision; no extra initial model or target-arm memory. New within-goal acceptance feedback may diverge as an outcome.",
              "failure_policy": "One revision attempt and one allocation per planned arm, no replacement. Revision failure remains a failed revision record and prevents claiming a completed policy comparison. Unknown worker failures stop confirmatory scoring; normal budget exhaustion stays in the denominator.",
              "future_policy": "All 18 allocations must terminate with intact evidence before any future metric; future metrics never select or promote policies.",
              "selection_policy": "Accepted incumbent, else first accepted, else independently replayed submission with highest development MSE skill and earlier-cycle tie break; score rejected valid submissions too; missing valid score gets utility -1, not an observed correlation.",
              "scope": SCOPE}
    write_json(args.output / "plan.json", {"design": design, "design_sha256": digest(design), "implementation": hashes})
    write_json(args.output / "shared_development_history.json", packet["development_history"])
    write_json(args.output / "parent_policy.json", packet["parent_policy"])
    copy_implementation(args.output, hashes)
    write_json(args.output / "report.json", {"status": "planned_waiting_for_revision", "planned": 18,
                                           "allocations": [], "plan_sha256": file_hash(args.output / "plan.json")})
    return {"status": "planned_waiting_for_revision", "planned": 18, "design_sha256": digest(design)}


def verify_plan(root):
    record = read(root / "plan.json")
    design = record["design"]
    require(digest(design) == record["design_sha256"], "Comparison plan changed")
    for name, sha in record["implementation"].items():
        require(file_hash(ROOT / name) == sha == file_hash(root / "implementation" / name), "Frozen comparison implementation changed")
    require(file_hash(Path(design["revision"]) / "protocol.json") == design["revision_protocol_sha256"], "Planned revision changed")
    _, packet = revision_context(design["revision"])
    require(read(root / "parent_policy.json") == packet["parent_policy"]
            and read(root / "shared_development_history.json") == packet["development_history"], "Shared starting state changed")
    return record


def materialize(root):
    root = root.resolve()
    with (root / ".materialize.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = verify_plan(root)
        require(read(root / "report.json")["status"] == "planned_waiting_for_revision", "Comparison preparation already attempted")
        design = record["design"]
        proposed, receipt = revision_evidence(design["revision"])
        require(proposed["parent_id"] == design["parent_id"]
                and proposed["history_sha256"] == design["history_sha256"], "Revision is not the planned successor")
        write_json(root / "report.json", {"status": "materializing", "planned": 18, "allocations": []})
        write_json(root / "candidate_policy.json", proposed["policy"])
        write_json(root / "revision_receipt.json", receipt)
        (root / "profiles").mkdir()
        for name, value in design["profiles"].items():
            write_json(root / "profiles" / (name + ".json"), value)
        arm_hashes, definitions = {}, {}
        for row in design["schedule"]:
            revised = row["condition"] == "revised_policy"
            arm = root / "arms" / row["id"]
            store = prepare_policy_goal(SimpleNamespace(
                output=arm, profile=root / "profiles" / (row["condition"] + ".json"), objective=design["objective"],
                family=row["case"], seed=row["seed"], condition=design["goal_condition"],
                improver_policy=root / ("candidate_policy.json" if revised else "parent_policy.json"),
                shared_history=root / "shared_development_history.json",
                **{k: Path(v) for k, v in design["sources"].items()}, **row["goal_budget"]))
            protocol = read(arm / "protocol.json")
            expected = design["partitions"][row["case"] + ":" + str(row["seed"])]
            require(all(protocol[k] == expected[k] for k in ["scope", "source", "split", "data_hashes"]), "Paired task data changed")
            protocol.update(implementation=record["implementation"], comparison_allocation=row,
                            scope_note=SCOPE, future_policy=design["future_policy"])
            copy_implementation(arm, record["implementation"])
            write_json(arm / "protocol.json", protocol)
            arm_hashes[row["id"]] = file_hash(arm / "protocol.json")
            definitions[row["id"]] = store.read()["definition_sha256"]
        write_json(root / "protocol.json", {"design": design, "design_sha256": digest(design),
                   "implementation": record["implementation"], "arm_protocols": arm_hashes,
                   "goal_definitions": definitions, "revision_receipt_sha256": file_hash(root / "revision_receipt.json"),
                   "candidate_policy_id": proposed["policy_id"], "plan_sha256": file_hash(root / "plan.json")})
        write_json(root / "report.json", {"status": "prepared", "planned": 18, "allocations": []})
        return {"status": "prepared", "planned": 18, "policy_id": proposed["policy_id"]}


def verify_campaign(root):
    verify_plan(root)
    protocol = goals.verify_protocol(root)
    require(protocol["plan_sha256"] == file_hash(root / "plan.json")
            and protocol["design"] == read(root / "plan.json")["design"], "Execution design differs from plan")
    proposed, receipt = revision_evidence(protocol["design"]["revision"])
    require(file_hash(root / "revision_receipt.json") == protocol["revision_receipt_sha256"]
            and read(root / "revision_receipt.json") == receipt
            and proposed["policy_id"] == protocol["candidate_policy_id"], "Revision receipt changed")
    for row in protocol["design"]["schedule"]:
        folder = root / "arms" / row["id"]
        p = read(folder / "protocol.json")
        state = GoalStore(folder).read()
        require(p["improver"]["policy_id"] == (proposed["policy_id"] if row["condition"] == "revised_policy" else proposed["parent_id"])
                and p["improver"]["shared_history_sha256"] == receipt["history_sha256"], "Policy or history differs across arms")
        actual = state["definition"]["goal"]
        require(all(actual[k] == v for k, v in row["goal_budget"].items())
                and state["definition"]["condition"] == "science_goal"
                and state["definition"]["profile"] == protocol["design"]["profiles"][row["condition"]], "Matched goal contract differs")
    return protocol


def contrasts(rows):
    table = {(r["case"], r["seed"], r["condition"]): r for r in rows}
    require(len(rows) == len(table) == 18, "Keep all 18 allocated outcomes")
    result = []
    for case, seed in sorted({k[:2] for k in table}):
        for left, right in [("revised_policy", "fixed_policy"), ("revised_policy", "fixed_policy_total_budget"),
                            ("fixed_policy_total_budget", "fixed_policy")]:
            a, b = table[(case, seed, left)], table[(case, seed, right)]
            result.append({"case": case, "seed": seed, "left": left, "right": right,
                           "complete_objective_difference": int(a["complete_objective"]) - int(b["complete_objective"]),
                           "ranking_utility_difference": a["ranking_utility"] - b["ranking_utility"],
                           "downstream_model_calls_difference": a["observed_model_calls"] - b["observed_model_calls"]})
    return result


def cost_summary(rows, revision):
    require(len(rows) == 18 and all(sum(r["condition"] == c for r in rows) == 6 for c in CONDITIONS), "All allocated costs required")
    groups = []
    for condition in CONDITIONS:
        selected = [r for r in rows if r["condition"] == condition]
        revise = condition == "revised_policy"
        groups.append({"condition": condition, "allocations": 6,
                       "complete_objectives": sum(r["complete_objective"] for r in selected),
                       "revision_model_calls": revision["observed_model_calls"] if revise else 0,
                       "revision_tool_calls": revision["observed_tool_calls"] if revise else 0,
                       "revision_elapsed_seconds": revision["host_wall_seconds"] if revise else 0,
                       "downstream_model_calls": sum(r["observed_model_calls"] for r in selected),
                       "downstream_tool_calls": sum(r["observed_tool_calls"] for r in selected),
                       "total_observed_model_calls": sum(r["observed_model_calls"] for r in selected) + (revision["observed_model_calls"] if revise else 0),
                       "total_observed_tool_calls": sum(r["observed_tool_calls"] for r in selected) + (revision["observed_tool_calls"] if revise else 0),
                       "total_charged_goal_wall_seconds": sum(r["charged_usage"]["wall_seconds"] for r in selected),
                       "cost_scope": "Revision shared once; downstream charged goal time and observed calls include failures. Allocated caps do not assert equal realized compute."})
    return groups


def score_offspring(root, rows):
    protocol = verify_campaign(root)
    analysis = goals.score_all(root, rows, contrasts=contrasts)
    for row in analysis["rows"]:
        executions = sorted((root / "arms" / row["id"] / "episodes").glob("*/execution"))
        row["token_usage"] = token_usage([path / "model_requests.jsonl" for path in executions])
        row["token_usage"]["complete"] = row["token_usage"]["complete"] and (
            row["token_usage"]["requests"] == row["observed_model_calls"] == row["charged_usage"]["model_calls"]
        )
    analysis.update(scope=SCOPE, revision_receipt=read(root / "revision_receipt.json"),
                    costs=cost_summary(analysis["rows"], read(root / "revision_receipt.json")),
                    parent_id=protocol["design"]["parent_id"], candidate_id=protocol["candidate_policy_id"],
                    promoted=False, recursive_effectiveness_verified=False)
    for cost in analysis["costs"]:
        records = [row["token_usage"] for row in analysis["rows"] if row["condition"] == cost["condition"]]
        if cost["condition"] == "revised_policy":
            records.append(analysis["revision_receipt"]["token_usage"])
        cost["reported_prompt_tokens"] = sum(record["prompt_tokens"] for record in records)
        cost["reported_completion_tokens"] = sum(record["completion_tokens"] for record in records)
        cost["token_usage_complete"] = all(record["complete"] for record in records)
    return analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "materialize", "run", "worker"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--credential-toml", type=Path)
    parser.add_argument("--allocation")
    args = parser.parse_args()
    if args.action == "plan":
        if not all([args.revision, args.profile, args.admission, args.features, args.baseline]):
            parser.error("Offline planning needs revision, profile and source paths")
        print(json.dumps(plan(args)))
    elif args.action == "materialize":
        print(json.dumps(materialize(args.output)))
    else:
        if not args.credential_toml or (args.action == "worker" and not args.allocation):
            parser.error("Execution requires credentials; workers also require allocation")
        verify_campaign(args.output.resolve())
        if args.action == "worker":
            asyncio.run(goals.worker(args))
        else:
            asyncio.run(goals.run(args, worker_script=Path(__file__), analyze=score_offspring))
