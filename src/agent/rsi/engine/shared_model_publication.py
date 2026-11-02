"""Native shared-goal provenance plus independent target-model publication.

A common execution is audited once for a batch publication. Target fits and
reuse are replayed independently. Synthetic executors cannot satisfy the native
proof gate, and a batch is never counted as multiple autonomous executions.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from pathlib import Path

from agent.science_goal import GoalSpec, GoalStore, goal_cycle_prompt, validate_assessment

from agent.rsi import model_tasks, shared_library_experience
from .context_feedback_dispatch import FIELD as FEEDBACK_FIELD
from .context_feedback_dispatch import policy_for
from .model_lifecycle import ModelLifecycle
from .protein_model import digest
from .shared_goal_history import SUPPORTED_POLICIES, make_prompt, project
from .shared_model_bundle import admit
from .shared_model_contracts import bundle, resolve_shared_result
from .shared_model_display import FIELD as DISPLAY_FIELD
from .storage import file_lock, safe_child, sha256_file
from .tool_contracts import validate_generated_result

SCHEMA = "shared_scientist_goal.v1"
CRITERIA = ("all_targets_accepted", "all_targets_freshly_reused")


from agent.rsi.layout import recorded_relative

def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    require(
        path.is_file() and not any(p.is_symlink() for p in [path, *path.parents]),
        "Missing or symlinked shared publication evidence",
    )
    value = json.loads(path.read_text())
    digest(value)
    return value


def _operation_receipts(folder, capability):
    """Bind shared scientific artifacts to the actual paired tool arguments."""
    events = [json.loads(x) for x in safe_child(folder, "execution/events.jsonl").read_text().splitlines()]
    pending, seen, completed = {}, set(), {}
    for event in events:
        identity = event.get("tool_call_id")
        if event.get("kind") == "tool_call_start":
            require(isinstance(identity, str) and identity not in seen, "Duplicate shared tool identity")
            pending[identity] = event
            seen.add(identity)
        elif event.get("kind") == "tool_result":
            require(identity in pending, "Unpaired shared tool result")
            call = pending.pop(identity)
            if event.get("is_error") is not False:
                continue
            name = call["tool_name"].removeprefix("mcp__venusfactory__")
            if name == "inspect_model_code_task":
                # Native harnesses may clip large inspection displays. Task/data
                # authority comes from the separately verified frozen capability,
                # not from parsing this informational display as a durable receipt.
                require(call["args"] == {}, "Shared inspection arguments changed")
                if DISPLAY_FIELD in read(capability / "task.json"):
                    value = json.loads(event["output"])
                    validate_generated_result(name, value, workspace=folder / "workspace", capability=capability)
                completed.setdefault(name, []).append({"observed": True})
                continue
            value = json.loads(event["output"])
            validate_generated_result(name, value, workspace=folder / "workspace", capability=capability)
            value = resolve_shared_result(name, value, workspace=folder / "workspace", capability=capability)
            if name == "evaluate_model_code":
                require(
                    hashlib.sha256(call["args"]["source_code"].encode()).hexdigest() == value["source_sha256"],
                    "Shared evaluated source differs from actual request",
                )
            elif name == "submit_model_code":
                require(
                    call["args"]["source_sha256"] == value["source_sha256"],
                    "Shared submitted source differs from actual request",
                )
                ledger = read(folder / "workspace/.shared_model_development.json")
                require(
                    call["args"]["rationale"] == ledger["submission"]["rationale"],
                    "Shared submission rationale changed",
                )
            elif name == "predict_with_generated_model":
                require(call["args"] == {"model_id": "all"}, "Shared reuse request changed")
            completed.setdefault(name, []).append(value)
    require(not pending, "Unfinished shared tool requests")
    return completed


def _capability(root, folder, protocol, selected, phase, records, data, profile):
    versions = safe_child(folder, "capability/bundles/versions")
    caps = list(versions.iterdir())
    require(len(caps) == 1, "Ambiguous shared capability")
    cap = caps[0]
    task, children = bundle(cap)
    ids = [t["id"] for t in selected]
    require(task["phase"] == phase and list(children) == ids, "Shared capability selection changed")
    for name, expected in task["implementation"].items():
        source = {"generated_tools.py": "shared_model_tools.py", "single_model_tools.py": "generated_tools.py"}.get(
            name, name
        )
        require(
            protocol["implementation"].get(recorded_relative(protocol["implementation"], source)) == expected,
            "Shared capability runtime differs from archived implementation",
        )
    allowed = set(task["implementation"]) | {"task.json", "manifest.json"}
    expected_inputs = {}
    for definition in selected:
        identity = definition["id"]
        child = children[identity]
        filename = "training.json" if phase == "develop" else "inputs.json"
        rows = (
            data[identity]["training"]
            if phase == "develop"
            else model_tasks.inference_inputs(data[identity]["future"], definition["plugin_id"])
        )
        expected = {
            "phase": phase,
            "scope": definition["scope"],
            "plugin_id": definition["plugin_id"],
            "inputs": {filename: digest(rows)},
            "shared_implementation_sha256": digest(task["implementation"]),
        }
        if phase == "develop":
            feedback = policy_for(
                {FEEDBACK_FIELD: protocol["target_training_baseline_feedback"], "scope": definition["scope"]},
                plugin_id=definition["plugin_id"],
            )
            if feedback != "none":
                expected[FEEDBACK_FIELD] = feedback
        else:
            packet = read(cap / "targets" / identity / "registered/capability.json")
            expected["capability_sha256"] = digest(packet)
            require(
                packet["model_id"] == records[identity]["model_id"], "Shared reuse capability substituted another model"
            )
            allowed.add(f"targets/{identity}/registered/capability.json")
        require(child == expected, "Shared target contract changed")
        allowed.add(f"targets/{identity}/task.json")
        relative = f"targets/{identity}/{filename}"
        require(read(folder / "workspace" / relative) == rows, "Shared exposed input differs from admitted partition")
        expected_inputs[relative] = sha256_file(folder / "workspace" / relative)
    require(
        {p.relative_to(cap).as_posix() for p in cap.rglob("*") if p.is_file()} == allowed,
        "Unexpected shared capability files",
    )
    lock = read(folder / "execution/profile.lock.json")
    require(
        lock["task"]["inputs"] == expected_inputs
        and lock["task"]["phase"] == phase
        and lock["task"]["mcp_entry"] == "generated_tools.py",
        "Shared harness input binding changed",
    )
    tools = (
        {"inspect_model_code_task", "evaluate_model_code", "submit_model_code"}
        if phase == "develop"
        else {"inspect_model_code_task", "discover_generated_models", "predict_with_generated_model"}
    )
    require(set(lock["task"]["tools"]) == tools, "Shared scientific tool permissions changed")
    manifest = read(cap / "manifest.json")
    require(manifest["metadata"]["model"] == profile["model"], "Shared capability model profile changed")
    return cap, lock


def _context(folder):
    from .scientist_model_library import _episodes

    root = Path(folder).absolute()
    require(not any(p.is_symlink() for p in [root, *root.parents]), "Symlinked shared goal")
    state, protocol = read(root / "goal.json"), read(root / "protocol.json")
    require(
        protocol["schema"] == SCHEMA
        and state["schema"] == 1
        and state["definition_sha256"] == digest(state["definition"]),
        "Invalid shared goal definition",
    )
    require(
        state["status"] in {"completed", "budget_exhausted"} and state["inflight"] is None,
        "Shared publication requires a terminal goal",
    )
    require(
        state["definition"]["condition"] == "science_goal"
        and tuple(state["definition"]["goal"]["criteria"]) == CRITERIA,
        "Invalid shared scientific criteria",
    )
    goal = GoalSpec(**{**state["definition"]["goal"], "criteria": tuple(state["definition"]["goal"]["criteria"])})
    complete = validate_assessment(GoalStore(root), state, state["assessment"])
    if state["status"] == "completed":
        require(complete, "Shared goal criteria do not pass")
    else:
        require(
            not complete
            and (
                len(state["cycles"]) >= goal.max_cycles
                or any(state["usage"][k] >= getattr(goal, k) for k in ["model_calls", "tool_calls", "wall_seconds"])
            ),
            "Shared exhaustion is not supported by the frozen budget",
        )
    require(bool(protocol["implementation"]), "Missing shared implementation archive")
    for name, expected in protocol["implementation"].items():
        require(
            sha256_file(safe_child(root, "implementation/" + name)) == expected,
            "Shared archived implementation changed",
        )
    tasks = [read(safe_child(root, "targets/" + t["id"] + "/model_task.json")) for t in protocol["targets"]]
    data = admit(tasks)
    history_policy = protocol["admission"]["shared_training_history"]
    require(history_policy in SUPPORTED_POLICIES, "Unsupported shared training history")
    admission = {
        "schema": SCHEMA,
        "tasks": [model_tasks.admission(t) for t in tasks],
        "training_baseline_feedback": protocol["target_training_baseline_feedback"],
        "implementation_sha256": digest(protocol["implementation"]),
        "shared_training_history": history_policy,
    }
    if shared_library_experience.FIELD in protocol:
        admission[shared_library_experience.FIELD] = protocol[shared_library_experience.FIELD]
    require(admission == protocol["admission"] == state["definition"]["admission"], "Shared admission changed")
    historical = shared_library_experience.verify_goal(root, state, protocol, tasks)
    for binding, task in zip(protocol["targets"], tasks, strict=True):
        identity = task["id"]
        require(binding["id"] == identity and binding["task_sha256"] == digest(task), "Shared task definition changed")
        require(
            model_tasks.source_and_split(task) == (binding["source"], binding["split"]),
            "Shared source/partition evidence changed",
        )
        for phase, rows in data[identity].items():
            require(
                read(root / "targets" / identity / "data" / (phase + ".json")) == rows
                and binding["data_hashes"][phase] == digest(rows),
                "Shared frozen cohort changed",
            )
    require(
        state["progress"]
        and set(state["progress"]) == {"targets"}
        and set(state["progress"]["targets"]) <= {t["id"] for t in tasks},
        "Shared goal omitted allocated targets or invented progress",
    )
    require(
        len({v["model_id"] for v in state["progress"]["targets"].values()}) == len(state["progress"]["targets"]),
        "Shared goal duplicated model identities",
    )
    # This is the unchanged native four-role/request/session audit, including its fixture-provider rejection.
    episodes, usage = _episodes(root, state, protocol)
    prefix = copy.deepcopy(state)
    prefix["cycles"], prefix["progress"] = [], {}
    records = {}
    for cycle in state["cycles"]:
        pending = [t for t in tasks if t["id"] in records and "reuse_receipt" not in records[t["id"]]]
        phase = "reuse" if pending else "develop"
        selected = pending if pending else [t for t in tasks if t["id"] not in records]
        require(
            selected and cycle["phase"] == phase and cycle["allocated_targets"] == [t["id"] for t in selected],
            "Shared episode changed target allocation or order",
        )
        episode = f"episodes/cycle_{cycle['index']:03d}"
        current = safe_child(root, episode)
        cap, lock = _capability(root, current, protocol, selected, phase, records, data, state["definition"]["profile"])
        history = project(root, prefix, selected, history_policy) if phase == "develop" else None
        if history is not None:
            require(
                read(current / "shared_training_history.json") == history
                and cycle["training_history_sha256"] == digest(history),
                "Shared training history changed",
            )
        else:
            require(
                "training_history_sha256" not in cycle and not (current / "shared_training_history.json").exists(),
                "Training history leaked into shared reuse",
            )
        prompt = shared_library_experience.append_prompt(
            make_prompt(goal_cycle_prompt(prefix), phase, history), phase, historical if phase == "develop" else None
        )
        shared_library_experience.verify_episode(root, prefix, protocol, cycle)
        require(
            read(current / "shared_prompt.json") == prompt and lock["task"]["prompt_sha256"] == digest(prompt),
            "Shared Scientist prompt changed",
        )
        if cycle["status"] == "completed":
            require(episode in episodes, "Shared episode lacks native Scientist proof")
            results = _operation_receipts(current, cap)
            expected_outcomes = {}
            if phase == "develop":
                require(bool(results.get("submit_model_code")), "No native shared submission")
                candidates = read(current / "workspace/candidate.json")["targets"]
                for candidate, definition in zip(candidates, selected, strict=True):
                    identity = definition["id"]
                    require(candidate["task_id"] == identity, "Shared target outcome reordered")
                    if candidate["status"] == "training_failed":
                        expected_outcomes[identity] = {"status": "training_failed"}
                        continue
                    controller = current / "targets" / identity / "controller"
                    policy = model_tasks.policy_for_task(definition, data[identity])
                    require(read(controller / "policy.json") == policy, "Shared controller policy changed")
                    lifecycle = ModelLifecycle(controller, policy)
                    local = lifecycle._state()
                    require(len(local["candidates"]) == 1, "Ambiguous shared controller candidates")
                    mid = next(iter(local["candidates"]))
                    model = lifecycle._model(mid)
                    report = read(controller / f"assessment_{mid}.json")
                    require(
                        type(report["passed"]) is bool
                        and digest(report) == local["candidates"][mid]["assessment_sha256"],
                        "Shared assessment record changed",
                    )
                    if report["passed"]:
                        report = lifecycle._verified_assessment(mid, local, model)
                    else:
                        require(
                            mid not in local["published"] and local["candidates"][mid]["status"] != "registered",
                            "Rejected shared target was registered",
                        )
                    require(
                        read(current / "targets" / identity / "acceptance.json") == report,
                        "Shared acceptance outcome changed",
                    )
                    require(
                        model == read(current / "workspace/targets" / identity / "candidate.json"),
                        "Shared controller substituted a different candidate",
                    )
                    expected_outcomes[identity] = {
                        "status": "accepted" if report["passed"] else "candidate_rejected",
                        "model_id": mid,
                    }
                    if report["passed"]:
                        records[identity] = {
                            "controller": controller.relative_to(root).as_posix(),
                            "model_id": mid,
                            "development_episode": episode,
                        }
            else:
                require(
                    bool(results.get("predict_with_generated_model"))
                    and bool(results.get("discover_generated_models")),
                    "No native shared reuse",
                )
                for definition in selected:
                    identity = definition["id"]
                    records[identity].update(
                        reuse_episode=episode, reuse_receipt=episode + "/receipts/" + identity + ".json"
                    )
                    expected_outcomes[identity] = {
                        "status": "freshly_reused",
                        "model_id": records[identity]["model_id"],
                    }
            require(cycle["target_outcomes"] == expected_outcomes, "Shared per-target outcomes changed")
        else:
            require(not cycle["target_outcomes"], "Failed shared episode claimed target success")
        require(
            cycle["progress"] == {"targets": records}, "Shared progress changed outside verified target transitions"
        )
        prefix["cycles"].append(cycle)
        prefix["progress"] = copy.deepcopy(cycle["progress"])
    require(state["progress"] == {"targets": records}, "Final shared progress differs from history")
    ready = {identity: row for identity, row in records.items() if "reuse_receipt" in row}
    expected_criteria = {
        CRITERIA[0]: {
            "passed": len(records) == len(tasks),
            "paths": {row["controller"] + "/assessment_" + row["model_id"] + ".json" for row in records.values()},
        },
        CRITERIA[1]: {"passed": len(ready) == len(tasks), "paths": {row["reuse_receipt"] for row in ready.values()}},
    }
    for name, expected in expected_criteria.items():
        check = state["assessment"]["criteria"][name]
        actual = check.get("evidence", [])
        require(
            check["passed"] == expected["passed"]
            and len(actual) == len(expected["paths"])
            and {x["path"] for x in actual} == expected["paths"],
            "Shared criterion coverage differs from qualified models",
        )
        require(
            all(sha256_file(safe_child(root, x["path"])) == x["sha256"] for x in actual),
            "Shared criterion evidence changed",
        )
    return root, state, protocol, tasks, data, episodes, usage, records


def validate_shared_goal(folder, model_id=None, *, replay_training=True):
    from .scientist_model_library import _model

    root, state, protocol, tasks, data, episodes, usage, records = _context(folder)
    ready = {identity: row for identity, row in records.items() if "reuse_receipt" in row}
    require(bool(ready), "No independently accepted and freshly reused shared target is available")
    if model_id is not None:
        require(
            model_id in {r["model_id"] for r in ready.values()},
            "Model is not a qualified and freshly reused output of this shared goal",
        )
    by_id = {t["id"]: t for t in protocol["targets"]}
    shared_id = digest(
        {"goal_sha256": sha256_file(root / "goal.json"), "protocol_sha256": sha256_file(root / "protocol.json")}
    )
    verified = []
    for task in tasks:
        identity = task["id"]
        if identity not in ready:
            continue
        record = ready[identity]
        if model_id is not None and record["model_id"] != model_id:
            continue
        development, reuse = record["development_episode"], record["reuse_episode"]
        require(
            record["controller"] == development + "/targets/" + identity + "/controller"
            and record["reuse_receipt"] == reuse + "/receipts/" + identity + ".json",
            "Shared target model layout changed",
        )
        paths = {
            "develop": development,
            "reuse": reuse,
            "development_workspace": development + "/workspace/targets/" + identity,
            "reuse_workspace": reuse + "/workspace/targets/" + identity,
            "receipt": record["reuse_receipt"],
        }
        result = _model(
            root,
            record,
            {"scope": task["scope"], "model_plugin_id": task["plugin_id"]},
            data[identity],
            episodes,
            task["minimum_skill"],
            replay_training,
            evidence_paths=paths,
        )
        verified.append(
            {
                **result,
                "origin": {
                    "goal_sha256": sha256_file(root / "goal.json"),
                    "protocol_sha256": sha256_file(root / "protocol.json"),
                    "model_id": record["model_id"],
                    "profile": state["definition"]["profile"],
                    "usage": usage,
                    "usage_attribution": "shared_goal_total_count_once",
                    "shared_execution_id": shared_id,
                    "shared_target_count": len(tasks),
                    "shared_published_target_count": len(ready),
                    "parent_goal_status": state["status"],
                    "stage": task["stage"],
                    "paper": task["paper"],
                    "task_id": identity,
                    "source": by_id[identity]["source"],
                    "reuse_receipt_sha256": result["reuse_receipt_sha256"],
                    "evidence_kind": "native_scientist_shared_program_model",
                    "future_labels_scored": False,
                },
            }
        )
    return verified[0] if model_id is not None else verified


def publish_shared_goal(library, folder):
    """Verify the common execution once, then publish separate target packets."""
    verified = validate_shared_goal(folder, replay_training=True)
    outcomes = [
        {
            "model_id": row["origin"]["model_id"],
            "task_id": row["origin"]["task_id"],
            "status": "published",
            **library._publish_verified(folder, row),
        }
        for row in verified
    ]
    report = {
        "schema": 1,
        "kind": "shared_model_goal_publication_attempt",
        "attempt_id": uuid.uuid4().hex,
        "origin_goal": str(folder.relative_to(library.owner)),
        "goal_sha256": sha256_file(folder / "goal.json"),
        "outcomes": outcomes,
        "all_published": True,
        "new_entries": sum(x["created"] for x in outcomes),
        "duplicate_entries": sum(not x["created"] for x in outcomes),
        "rejected_entries": 0,
        "new_autonomous_executions": 0,
        "shared_execution_id": verified[0]["origin"]["shared_execution_id"],
        "verified_shared_goals": 1,
        "verified_target_scopes": len(outcomes),
        "shared_usage": verified[0]["origin"]["usage"],
    }
    state, protocol = read(folder / "goal.json"), read(folder / "protocol.json")
    published = {row["task_id"] for row in outcomes}
    records = state["progress"]["targets"]
    report.update(
        parent_goal_status=state["status"],
        allocated_scopes=len(protocol["targets"]),
        all_allocated_scopes_published=len(published) == len(protocol["targets"]),
        unpublished_targets=[
            {
                "task_id": task["id"],
                "status": "accepted_without_fresh_reuse" if task["id"] in records else "not_accepted_within_budget",
            }
            for task in protocol["targets"]
            if task["id"] not in published
        ],
    )
    with file_lock(library.root / ".library.lock"):
        identity = library._immutable("publication_attempts", report)
    return {**report, "report_id": identity}
