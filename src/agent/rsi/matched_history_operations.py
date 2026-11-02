"""Observed resource use and verified training-history exposure, without labels."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from . import shared_goal_history
from .matched_history_analysis import ARMS, CONTRASTS, interval, require
from .protein_model import digest
from .shared_model_contracts import history_path, resolve_shared_result
from .tool_contracts import validate_generated_result

TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
HISTORY_MARKER = "Prior paired TRAINING-tool observations from this goal follow as JSON."


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def request_counts(records, *, model):
    """Count physical requests once; absent usage remains unknown, never zero."""
    requests, responses = {}, {}
    other = Counter()
    for record in records:
        event, call = record.get("event"), record.get("call")
        if event == "request":
            require(type(call) is int and call > 0 and call not in requests, "Duplicate/invalid model request")
            require(record["model"] == model, "Unexpected model in frozen unit")
            requests[call] = record
        elif event == "response":
            require(
                type(call) is int and call in requests and call not in responses, "Unpaired/duplicate model response"
            )
            responses[call] = record
        else:
            other[str(event)] += 1
    tokens = {}
    for field in TOKEN_FIELDS:
        values = []
        for call in requests:
            usage = responses.get(call, {}).get("usage")
            value = usage.get(field) if isinstance(usage, dict) else None
            require(value is None or (type(value) is int and value >= 0), "Invalid logged token count")
            if value is not None:
                values.append(value)
        tokens[field] = {
            "total": sum(values) if len(values) == len(requests) else None,
            "known_sum": sum(values),
            "known_requests": len(values),
            "unknown_requests": len(requests) - len(values),
        }
    durations = [r.get("wall_seconds") for r in responses.values()]
    require(all(v is None or numeric(v) for v in durations), "Invalid response duration")
    known = [v for v in durations if v is not None]
    return {
        "requests": len(requests),
        "responses": len(responses),
        "unanswered_requests": len(requests) - len(responses),
        "completed_responses": sum(r.get("completed") is True for r in responses.values()),
        "http_statuses": dict(Counter(str(r.get("status")) for r in responses.values())),
        "other_events": dict(other),
        "tokens": tokens,
        "response_wall_seconds": math.fsum(known) if len(known) == len(requests) else None,
        "known_response_wall_seconds": math.fsum(known),
    }


def tool_pairs(events):
    """Pair normalized calls, retaining failed and interrupted invocations."""
    seen, pending, pairs, names = set(), {}, [], Counter()
    for event in events:
        identity = event.get("tool_call_id")
        if event.get("kind") == "tool_call_start":
            require(isinstance(identity, str) and identity and identity not in seen, "Duplicate/invalid tool call")
            seen.add(identity)
            pending[identity] = event
            names[event["tool_name"].removeprefix("mcp__venusfactory__")] += 1
        elif event.get("kind") == "tool_result":
            require(identity in pending, "Unpaired/duplicate tool result")
            pairs.append((pending.pop(identity), event))
    return {
        "started": len(seen),
        "returned": len(pairs),
        "unreturned": len(pending),
        "error_results": sum(e.get("is_error") is not False for _, e in pairs),
        "by_tool": dict(names),
    }, pairs


def check_history(packet, *, expected, prompt, cycle, policy):
    """The exact verified history must appear in the prompt bound to execution."""
    require(packet == expected, "Training-history projection differs")
    if expected is None:
        require(
            "training_history_sha256" not in cycle and HISTORY_MARKER not in prompt,
            "History leaked into excluded episode",
        )
        return {"source_hashes": [], "sources": 0, "bytes": 0, "has_programs": False}
    require(cycle["phase"] == "develop" and expected["policy"] == policy, "History phase/policy differs")
    require(cycle["training_history_sha256"] == digest(expected), "History checkpoint hash differs")
    serialized = json.dumps(expected, ensure_ascii=False)
    require(HISTORY_MARKER in prompt and prompt.endswith(serialized), "Exact history missing from supplied prompt")
    size = len(serialized.encode("utf-8"))
    require(size <= 48 * 1024 and len(expected["attempts"]) <= 2, "History source or byte cap exceeded")
    hashes = []
    for attempt in expected["attempts"]:
        sha = hashlib.sha256(attempt["source_code"].encode()).hexdigest()
        require(sha == attempt["source_sha256"] and sha not in hashes, "History source identity duplicated or changed")
        require(
            type(attempt["episode"]) is int and 0 <= attempt["episode"] < cycle["index"],
            "History includes a non-prior episode",
        )
        hashes.append(sha)
    return {"source_hashes": hashes, "sources": len(hashes), "bytes": size, "has_programs": bool(hashes)}


def aggregate_requests(items):
    total = {
        key: sum(item[key] for item in items)
        for key in ("requests", "responses", "unanswered_requests", "completed_responses")
    }
    for key in ("http_statuses", "other_events"):
        counts = Counter()
        for item in items:
            counts.update(item[key])
        total[key] = dict(counts)
    total["tokens"] = {}
    for field in TOKEN_FIELDS:
        entries = [item["tokens"][field] for item in items]
        unknown = sum(item["unknown_requests"] for item in entries)
        known_sum = sum(item["known_sum"] for item in entries)
        total["tokens"][field] = {
            "total": None if unknown else known_sum,
            "known_sum": known_sum,
            "known_requests": sum(item["known_requests"] for item in entries),
            "unknown_requests": unknown,
        }
    total["known_response_wall_seconds"] = math.fsum(item["known_response_wall_seconds"] for item in items)
    total["response_wall_seconds"] = (
        None if any(item["response_wall_seconds"] is None for item in items) else total["known_response_wall_seconds"]
    )
    return total


def resource_accounting(campaign, terminal, state, requests, tools):
    """Failed reservations remain charges, never fabricated physical usage."""
    failed = terminal.get("unpublished_failure") is True
    if failed:
        require(terminal.get("classification") == "unpublished_failure", "Missing failure classification")
        require(all(b["status"] == "failed" for b in campaign["batches"].values()), "Mixed failed accounting")
        require(
            all(a.get("publication") is None for b in campaign["batches"].values() for a in b["attempts"]),
            "Failed accounting contains publication",
        )
        require(
            requests == state["usage"]["model_calls"] and tools == state["usage"]["tool_calls"],
            "Failed goal observed usage differs",
        )
        return {
            "campaign_wall_seconds": None,
            "charged_usage": campaign["usage"],
            "campaign_usage_known": False,
            "usage_basis": "independently_replayed_episode_logs",
        }
    require(requests == terminal["usage"]["model_calls"], "Unit request accounting differs")
    require(tools == terminal["usage"]["tool_calls"], "Unit tool accounting differs")
    return {
        "campaign_wall_seconds": terminal["usage"]["wall_seconds"],
        "campaign_usage_known": True,
        "usage_basis": "independently_replayed_episode_logs",
    }


def duration_totals(values):
    durations = [u["campaign_wall_seconds"] for u in values]
    known = [v for v in durations if v is not None]
    return {
        "campaign_wall_seconds": math.fsum(known) if len(known) == len(durations) else None,
        "known_campaign_wall_seconds": math.fsum(known),
        "unknown_campaign_wall_units": len(durations) - len(known),
    }


def collect_unit(evidence, plan, context):
    """Audit closed-unit operational records; this function never opens labels."""
    folder, campaign, terminal = context
    require(
        campaign["owner"] is None
        and campaign["inflight"] is None
        and campaign["status"] in {"completed", "needs_tasks", "budget_exhausted"},
        "Operational audit requires a closed unit",
    )
    batch = plan["batches"][0]
    require(len(plan["batches"]) == 1, "Single shared batch required")
    goal = evidence.path(folder / "campaign/goals" / batch["id"], inside=folder)
    state, protocol = evidence.read(goal / "goal.json"), evidence.read(goal / "protocol.json")
    require(state["inflight"] is None and state["status"] in {"completed", "budget_exhausted"}, "Goal still running")
    policy = batch["shared_training_history"]
    require(protocol["admission"]["shared_training_history"] == policy, "Goal history policy differs")
    tasks = {t["id"]: t for t in batch["tasks"]}
    require(
        {t["id"]: t["task_sha256"] for t in protocol["targets"]} == {tid: digest(t) for tid, t in tasks.items()},
        "Goal task identities differ",
    )
    # These functions inspect only training-tool feedback and capability artifacts.
    for name in (
        "shared_goal_history.py",
        "shared_history_selection.py",
        "shared_model_contracts.py",
        "tool_contracts.py",
        "training_feedback_view.py",
        "protein_model.py",
        "storage.py",
        "versions.py",
    ):
        relative = "src/agent/rsi/" + name
        expected = protocol["implementation"][relative]
        evidence.read_bytes(goal / "implementation" / relative, expected)
        evidence.read_bytes(Path(__file__).with_name(name), expected)
    require([c["index"] for c in state["cycles"]] == list(range(len(state["cycles"]))), "Cycle ordering differs")
    cycles, prefix = [], {**state, "cycles": []}
    for cycle in state["cycles"]:
        require(evidence.read(goal / f"cycle_{cycle['index']:03d}.json") == cycle, "Cycle checkpoint changed")
        episode = goal / "episodes" / f"cycle_{cycle['index']:03d}"
        require(
            cycle["run_record"] == f"episodes/cycle_{cycle['index']:03d}/execution/run.json", "Run record path differs"
        )
        audit = evidence.read(episode / "lifecycle_execution_audit.json")
        require(digest(audit) == cycle["execution_audit_sha256"], "Lifecycle audit changed")
        run = evidence.read(episode / "execution/run.json")
        lock = evidence.read(episode / "execution/profile.lock.json")
        prompt = evidence.read(episode / "shared_prompt.json")
        require(
            isinstance(prompt, str) and lock["task"]["prompt_sha256"] == digest(prompt),
            "Execution prompt binding differs",
        )
        require(run["profile_sha256"] == lock["sha256"], "Run profile binding differs")
        selected = [tasks[tid] for tid in cycle["allocated_targets"]]
        expected = shared_goal_history.project(goal, prefix, selected, policy) if cycle["phase"] == "develop" else None
        hp = evidence.path(episode / "shared_training_history.json")
        require(hp.exists() == (expected is not None), "Unexpected/missing history packet")
        history = check_history(
            evidence.read(hp) if hp.exists() else None, expected=expected, prompt=prompt, cycle=cycle, policy=policy
        )
        requests = request_counts(
            [
                json.loads(s)
                for s in evidence.read_bytes(episode / "execution/model_requests.jsonl").decode().splitlines()
                if s
            ],
            model=batch["profile"]["model"]["model"],
        )
        history["execution_made_model_requests"] = requests["requests"] > 0
        events = [
            json.loads(s) for s in evidence.read_bytes(episode / "execution/events.jsonl").decode().splitlines() if s
        ]
        tools, pairs = tool_pairs(events)
        caps = list(evidence.path(episode / "capability/bundles/versions").iterdir())
        require(len(caps) == 1, "Ambiguous capability version")
        for relative, sha in lock["task"]["capabilities"].items():
            evidence.read_bytes(caps[0] / relative, sha, inside=caps[0])
        numerical_calls, numerical_targets = 0, 0
        for call, result in pairs:
            if (
                call["tool_name"].removeprefix("mcp__venusfactory__") == "evaluate_model_code"
                and result.get("is_error") is False
            ):
                value = json.loads(result["output"])
                validate_generated_result(
                    "evaluate_model_code", value, workspace=episode / "workspace", capability=caps[0]
                )
                full = resolve_shared_result(
                    "evaluate_model_code", value, workspace=episode / "workspace", capability=caps[0]
                )
                if "artifact" in value:
                    artifact = history_path(episode / "workspace", "shared_tool_response_", value["result_sha256"])
                    require(
                        evidence.read(artifact, inside=episode / "workspace") == full,
                        "Training result artifact changed",
                    )
                require(
                    full["source_sha256"] == hashlib.sha256(call["args"]["source_code"].encode()).hexdigest(),
                    "Training evaluation source differs",
                )
                require(
                    len({t["task_id"] for t in full["results"]}) == len(full["results"]), "Repeated training result"
                )
                require(
                    {t["task_id"] for t in full["results"]} == set(cycle["allocated_targets"]),
                    "Training evaluation target coverage differs",
                )
                numerical_calls += 1
                numerical_targets += len(full["results"])
        require(
            requests["requests"] == cycle["usage"]["model_calls"] == run["llm_calls"],
            "Recorded request accounting differs",
        )
        require(
            tools["started"] == cycle["usage"]["tool_calls"] == run["tool_calls"], "Recorded tool accounting differs"
        )
        require(numeric(cycle["usage"]["wall_seconds"]) and numeric(run["host_wall_seconds"]), "Missing wall duration")
        cycles.append(
            {
                "index": cycle["index"],
                "phase": cycle["phase"],
                "status": cycle["status"],
                "requests": requests,
                "tools": tools,
                "training_evaluation_calls_with_results": numerical_calls,
                "target_training_results": numerical_targets,
                "checkpoint_wall_seconds": cycle["usage"]["wall_seconds"],
                "host_execution_wall_seconds": run["host_wall_seconds"],
                "history": history,
            }
        )
        prefix["cycles"].append(cycle)
    require(cycles, "No closed episodes to audit")
    publications = [
        r
        for b in campaign["batches"].values()
        for a in b["attempts"]
        for r in (a.get("publication") or {}).get("outcomes", [])
    ]
    require(
        len({r["task_id"] for r in publications}) == len(publications)
        and {r["task_id"] for r in publications} <= set(tasks),
        "Invalid publication coverage",
    )
    requests = aggregate_requests([c["requests"] for c in cycles])
    accounting = resource_accounting(
        campaign, terminal, state, requests["requests"], sum(c["tools"]["started"] for c in cycles)
    )
    development = [c for c in cycles if c["phase"] == "develop"]
    result = {
        "allocated": len(tasks),
        "registered": len(publications),
        "cycles": cycles,
        "requests": requests,
        "tool_calls": sum(c["tools"]["started"] for c in cycles),
        "training_evaluation_calls_with_results": sum(c["training_evaluation_calls_with_results"] for c in cycles),
        "target_training_results": sum(c["target_training_results"] for c in cycles),
        "checkpoint_wall_seconds": math.fsum(c["checkpoint_wall_seconds"] for c in cycles),
        "host_execution_wall_seconds": math.fsum(c["host_execution_wall_seconds"] for c in cycles),
        **accounting,
        "history": {
            "development_cycles": len(development),
            "exposed_development_cycles": sum(
                c["history"]["has_programs"] and c["history"]["execution_made_model_requests"] for c in development
            ),
            "source_occurrences": sum(c["history"]["sources"] for c in development),
            "distinct_source_programs": len({s for c in development for s in c["history"]["source_hashes"]}),
            "packet_utf8_bytes": sum(c["history"]["bytes"] for c in development),
        },
        "history_projection_replayed": True,
        "final_labels_read": False,
        "external_model_calls": 0,
    }
    return result


def cost_values(unit):
    return {
        "model_requests": unit["requests"]["requests"],
        "total_tokens": unit["requests"]["tokens"]["total_tokens"]["total"],
        "tool_calls": unit["tool_calls"],
        "training_evaluation_calls_with_results": unit["training_evaluation_calls_with_results"],
        "target_training_results": unit["target_training_results"],
        "checkpoint_wall_seconds": unit["checkpoint_wall_seconds"],
        "host_execution_wall_seconds": unit["host_execution_wall_seconds"],
        "campaign_wall_seconds": unit["campaign_wall_seconds"],
    }


def summarize_operations(design, units):
    """Match resource contrasts by assigned block, including zero-acquisition units."""
    require(
        len({u["id"] for u in design["units"]}) == len(design["units"])
        and set(units) == {u["id"] for u in design["units"]},
        "Operational unit coverage differs",
    )
    grouped, blocks = defaultdict(list), defaultdict(dict)
    for assignment in design["units"]:
        unit = units[assignment["id"]]
        require(assignment["arm"] in ARMS, "Unknown operational arm")
        require(
            unit["allocated"] == design["tasks_per_block"] and 0 <= unit["registered"] <= unit["allocated"],
            "Operational denominator differs",
        )
        require(
            all(value is None or numeric(value) for value in cost_values(unit).values()), "Invalid operational cost"
        )
        key = (assignment["project"], assignment["source_block"])
        require(assignment["arm"] not in blocks[key], "Repeated operational arm/block")
        blocks[key][assignment["arm"]] = unit
        grouped[assignment["arm"]].append(unit)
    require(
        set(grouped) == set(ARMS) and all(set(b) == set(ARMS) for b in blocks.values()), "Incomplete operational block"
    )
    arms = {}
    for arm, values in grouped.items():
        allocated, registered = sum(u["allocated"] for u in values), sum(u["registered"] for u in values)
        requests = aggregate_requests([u["requests"] for u in values])
        tokens = requests["tokens"]["total_tokens"]["total"]
        arms[arm] = {
            "units": len(values),
            "allocated": allocated,
            "registered": registered,
            "requests": requests,
            "requests_per_assigned_task": requests["requests"] / allocated,
            "requests_per_registered_model": requests["requests"] / registered if registered else None,
            "total_tokens_per_registered_model": tokens / registered if registered and tokens is not None else None,
            "tool_calls": sum(u["tool_calls"] for u in values),
            "training_evaluation_calls_with_results": sum(u["training_evaluation_calls_with_results"] for u in values),
            "target_training_results": sum(u["target_training_results"] for u in values),
            **duration_totals(values),
            "history": {
                "exposed_units": sum(u["history"]["exposed_development_cycles"] > 0 for u in values),
                "unexposed_units": sum(u["history"]["exposed_development_cycles"] == 0 for u in values),
                **{
                    key: sum(u["history"][key] for u in values)
                    for key in (
                        "development_cycles",
                        "exposed_development_cycles",
                        "source_occurrences",
                        "packet_utf8_bytes",
                    )
                },
            },
        }
    contrasts = {}
    for left, right in CONTRASTS:
        effects = {}
        for metric in cost_values(next(iter(units.values()))):
            values, missing = [], []
            for (project, block), assigned in sorted(blocks.items()):
                a, b = cost_values(assigned[left])[metric], cost_values(assigned[right])[metric]
                if a is None or b is None:
                    missing.append({"project": project, "block": block})
                else:
                    values.append((project, block, (a - b) / assigned[left]["allocated"]))
            # Do not compare cheaper-looking units by dropping missing cost logs.
            summary = (
                interval(values) if not missing else {"estimate": None, "interval_95": None, "blocks": len(values)}
            )
            effects[metric + "_per_assigned_task"] = {
                **summary,
                "missing_blocks": missing,
                "coverage_complete": not missing,
                "direction": "negative_favors_left",
            }
        contrasts[left + "_minus_" + right] = effects
    return {
        "schema": "matched_history_operations.v1",
        "arms": arms,
        "paired_cost_contrasts": contrasts,
        "units": units,
        "request_counting": "Physical request records, including failures and unanswered requests; reservations are not requests.",
        "token_counting": "Provider-reported usage only; unknown request usage is retained and prevents a complete total.",
        "numerical_counting": "Evaluate-tool returns and per-target training result records; these are not counts of newly acquired models or all internal estimator fits.",
        "duration_counting": "Gateway response time, host execution time, cycle checkpoint time and campaign wall time remain separate; they overlap and are not added together.",
        "history_counting": "Exact verified training-program packets bound to execution prompts. Exposed cycles also require an observed model request; packet bytes and source occurrences describe recorded prompts. Exposure alone does not establish useful reuse or provider-level causal utilization.",
        "zero_exposure_units_retained_in_assigned_arms": True,
        "final_labels_read": False,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
