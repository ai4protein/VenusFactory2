"""Persistent multi-task Scientist campaigns with publication-based completion."""

from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

from agent.harness.contracts import Profile, digest, file_hash
from agent.harness.native_expert_team import NATIVE_EXPERT_HARNESSES
from agent.science_goal import GoalSpec, GoalStore

from . import context_feedback_dispatch as context_feedback
from . import context_transfer, library_experience, model_tasks, training_attempt_memory
from .model_coverage import PROGRAM_ENTRIES, SCIENTIFIC_SCOPES, counting_policy, resolve_scope, summarize_scopes
from .scientist_model_library import ScientistModelLibrary
from .storage import atomic_json, file_lock, safe_child

RESOURCES = ("model_calls", "tool_calls", "wall_seconds")
TERMINAL = {"completed", "publication_rejected", "budget_exhausted", "failed", "interrupted"}
COVERAGE_TARGET = "coverage_target.v1"
ALL_ALLOCATIONS = "all_allocations.v1"


def execution_policy(definition):
    """Missing policy retains historical acquisition stopping behavior."""
    value = definition.get("execution_policy", COVERAGE_TARGET)
    if value not in (COVERAGE_TARGET, ALL_ALLOCATIONS):
        raise ValueError("Unsupported campaign execution policy")
    return value


def allocation_execution(state):
    """Coverage success and completion of predeclared executions are separate."""
    unfinished = [
        name
        for name, item in state["allocations"].items()
        if item["status"] not in TERMINAL | {"cancelled"} or not item["attempts"]
    ]
    return {
        "execution_policy": execution_policy(state["definition"]),
        "required_allocations": len(state["allocations"]),
        "terminal_attempted_allocations": len(state["allocations"]) - len(unfinished),
        "unfinished_allocations": unfinished,
        "execution_complete": not unfinished and state["inflight"] is None,
    }


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require(path.is_file() and not path.is_symlink(), "Missing or symlinked campaign evidence")
    return json.loads(path.read_text())


def budget(value):
    require(set(value) == set(RESOURCES), "Invalid campaign budget fields")
    for key, n in value.items():
        require(type(n) in (int, float) and math.isfinite(n) and n >= 0, "Invalid campaign resource budget")
        if key != "wall_seconds":
            require(type(n) is int, "Call budgets must be integers")
    return dict(value)


def process_identity():
    return {
        "pid": os.getpid(),
        "start_ticks": Path(f"/proc/{os.getpid()}/stat").read_text().split()[21],
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }


def process_live(identity):
    if not identity:
        return False
    path = Path(f"/proc/{identity['pid']}/stat")
    try:
        fields = path.read_text().split()
        return (
            fields[2] != "Z"
            and fields[21] == identity["start_ticks"]
            and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == identity["boot_id"]
        )
    except FileNotFoundError:
        return False


def admit_allocations(allocations, *, policy=PROGRAM_ENTRIES):
    counting_policy({"counting_policy": policy})
    require(isinstance(allocations, list) and 1 <= len(allocations) <= 10000, "Declare 1..10000 task allocations")
    result, seen, coverage_cache = [], set(), {}
    for item in allocations:
        required = {"id", "task", "profile", "generations", "goal_budget"}
        require(
            required
            <= set(item)
            <= required | {"experience", "coverage", context_feedback.FIELD, training_attempt_memory.FIELD},
            "Invalid task allocation",
        )
        if "experience" in item:
            library_experience.campaign_policy(item["experience"])
        name = item["id"]
        require(
            isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", name) and name not in seen,
            "Task allocation IDs must be unique safe names",
        )
        seen.add(name)
        model_tasks.load_task(item["task"])
        if "experience" in item:
            context_transfer.validate_declaration(item["experience"], plugin_id=item["task"]["plugin_id"])
        context_feedback.policy_for(item, plugin_id=item["task"]["plugin_id"])
        training_attempt_memory.policy_for(item)
        profile = Profile.from_dict(item["profile"])
        require(
            profile.mode == "science_expert" and profile.harness in NATIVE_EXPERT_HARNESSES,
            "A Scientist harness with a publication auditor is required",
        )
        require(type(item["generations"]) is int and 1 <= item["generations"] <= 3, "Supported lineage length is 1..3")
        spec = GoalSpec("Admitted campaign task", ("accepted_model", "fresh_session_reuse"), **item["goal_budget"])
        require(spec.max_cycles >= 2 * item["generations"], "Allow development and fresh reuse for each generation")
        if policy == SCIENTIFIC_SCOPES:
            resolve_scope(item, cache=coverage_cache)
        result.append(copy.deepcopy(item))
    return result


class ModelCampaignStore:
    def __init__(self, root):
        root = Path(root).absolute()
        require(not any(p.is_symlink() for p in [root, *root.parents]), "Campaign paths cannot traverse symlinks")
        self.root = root.resolve()

    @classmethod
    def create(
        cls,
        root,
        *,
        objective,
        target_models,
        library_root,
        repository,
        implementation,
        allocations,
        resources,
        required_stages=(),
        counting=SCIENTIFIC_SCOPES,
        execution=COVERAGE_TARGET,
    ):
        require(isinstance(objective, str) and objective.strip(), "Explicit research objective required")
        require(type(target_models) is int and target_models > 0, "Positive verified-model target required")
        require(
            isinstance(required_stages, (list, tuple))
            and len(set(required_stages)) == len(required_stages)
            and all(s in model_tasks.STAGES for s in required_stages),
            "Invalid required task stages",
        )
        counting_policy({"counting_policy": counting})
        execution_policy({"execution_policy": execution})
        allocations = admit_allocations(allocations, policy=counting)
        resources = budget(resources)
        store = cls(root)
        library = ScientistModelLibrary(library_root)
        library.goal(store.root)
        repository = Path(repository).resolve()
        require(bool(implementation), "Frozen execution source is required")
        for name, sha in implementation.items():
            require(file_hash(safe_child(repository, name)) == sha, "Campaign source changed before allocation")
        definition = {
            "schema": 1,
            "objective": objective,
            "target_models": target_models,
            "library_root": str(library.root),
            "repository": str(repository),
            "implementation": implementation,
            "required_stages": list(required_stages),
        }
        if execution != COVERAGE_TARGET:
            definition["execution_policy"] = execution
        if counting != PROGRAM_ENTRIES:
            definition["counting_policy"] = counting
        store.root.mkdir(parents=True, exist_ok=False)
        state = {
            "schema": 1,
            "definition": definition,
            "definition_sha256": digest(definition),
            "status": "ready",
            "plans": [],
            "allocations": {},
            "limits": dict.fromkeys(RESOURCES, 0),
            "usage": dict.fromkeys(RESOURCES, 0),
            "inflight": None,
            "owner": None,
            "events": [],
            "verified_entry_ids": [],
            "last_publication_audit": None,
        }
        atomic_json(store.root / "campaign.json", state)
        store.extend(allocations, resources=resources)
        return store

    def read(self, *, verify_source=True):
        state = read(self.root / "campaign.json")
        require(
            state["schema"] == 1 and digest(state["definition"]) == state["definition_sha256"],
            "Campaign definition changed",
        )
        counting_policy(state["definition"])
        execution_policy(state["definition"])
        allocations, limits, previous = {}, dict.fromkeys(RESOURCES, 0), None
        for identity in state["plans"]:
            require(re.fullmatch(r"[0-9a-f]{64}", identity) is not None, "Invalid plan identity")
            plan = read(safe_child(self.root, f"plans/{identity}.json"))
            require(digest(plan) == identity and plan["previous"] == previous, "Campaign plan history changed")
            for item in plan["allocations"]:
                require(item["id"] not in allocations, "Duplicate allocation in plan history")
                allocations[item["id"]] = item
            for k, n in budget(plan["resources"]).items():
                limits[k] += n
            previous = identity
        require(
            limits == state["limits"] and set(allocations) == set(state["allocations"]), "Campaign inventory changed"
        )
        budget(state["usage"])
        if verify_source:
            repository = Path(state["definition"]["repository"])
            for name, sha in state["definition"]["implementation"].items():
                require(
                    file_hash(safe_child(repository, name)) == sha,
                    "Campaign implementation changed; use its frozen repository",
                )
        return state

    def specifications(self, state=None):
        state = state or self.read()
        return {
            a["id"]: a
            for identity in state["plans"]
            for a in read(self.root / "plans" / (identity + ".json"))["allocations"]
        }

    def mutate(self, callback, *, verify_source=True):
        with file_lock(self.root / ".state.lock"):
            state = self.read(verify_source=verify_source)
            callback(state)
            atomic_json(self.root / "campaign.json", state)
            return state

    @contextmanager
    def lease(self):
        with (self.root / ".runner.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError("A runner already owns this campaign") from exc
            yield

    def extend(self, allocations, *, resources):
        allocations = admit_allocations(allocations, policy=counting_policy(self.read()["definition"]))
        resources = budget(resources)
        with self.lease():

            def change(state):
                require(
                    state["inflight"] is None
                    and state["status"] in {"ready", "paused", "needs_tasks", "budget_exhausted"},
                    "Extend a quiescent campaign",
                )
                require(
                    not {a["id"] for a in allocations} & set(state["allocations"]),
                    "Existing allocations cannot be replaced",
                )
                plan = {
                    "schema": 1,
                    "previous": state["plans"][-1] if state["plans"] else None,
                    "allocations": allocations,
                    "resources": resources,
                }
                identity = digest(plan)
                path = safe_child(self.root, f"plans/{identity}.json")
                if path.exists():
                    require(read(path) == plan, "Uncommitted plan object changed")
                else:
                    atomic_json(path, plan)
                    path.chmod(0o444)
                state["plans"].append(identity)
                for item in allocations:
                    state["allocations"][item["id"]] = {"status": "queued", "attempts": [], "entries": []}
                for k, n in resources.items():
                    state["limits"][k] += n
                if state["status"] != "paused":
                    state["status"] = "ready"
                state["events"].append({"event": "plan_added", "plan": identity})

            return self.mutate(change)

    def goal(self, identity):
        require(identity in self.read()["allocations"], "Unknown task allocation")
        return safe_child(self.root, "goals/" + identity)

    def pause(self):
        def change(state):
            require(state["status"] != "completed", "Completed campaign cannot be paused")
            state["status"] = "paused"
            state["events"].append({"event": "pause_requested"})

        state = self.mutate(change, verify_source=False)
        if state["inflight"]:
            goal = safe_child(self.root, "goals/" + state["inflight"]["allocation"])
            if (goal / "goal.json").is_file():
                GoalStore(goal).pause()
        return state

    def resume(self):
        with self.lease():
            prior = self.read()
            require(
                prior["status"] in {"paused", "waiting_for_model"} and prior["inflight"] is None,
                "Resume requires a quiescent paused campaign or explicit model-access recovery",
            )
            for name, item in prior["allocations"].items():
                if item["status"] in {"paused", "waiting_for_model"}:
                    child = GoalStore(self.goal(name))
                    if child.read()["status"] in {"paused", "waiting_for_input"}:
                        child.resume()
                    else:
                        require(child.read()["status"] == "active", "Child goal is not resumable")

            def change(state):
                require(
                    state["status"] in {"paused", "waiting_for_model"} and state["inflight"] is None,
                    "Resume requires a quiescent paused campaign or explicit model-access recovery",
                )
                for record in state["allocations"].values():
                    if record["status"] in {"paused", "waiting_for_model"}:
                        record["status"] = "queued"
                state["status"] = "ready"
                state["events"].append({"event": "resume_requested"})

            return self.mutate(change)

    def recover(self):
        """Retain an unknown reservation. Never restart the interrupted allocation."""
        with self.lease():

            def change(state):
                require(state["inflight"] is not None, "No interrupted reservation to recover")
                require(not process_live(state["inflight"]["owner"]), "The recorded process is still alive")
                name = state["inflight"]["allocation"]
                state["allocations"][name]["status"] = "interrupted"
                state["allocations"][name]["attempts"].append(
                    {"status": "interrupted", "usage_known": False, "charged": state["inflight"]["reservation"]}
                )
                state["events"].append({"event": "interrupted_reservation_retained", "allocation": name})
                state["inflight"], state["owner"], state["status"] = None, None, "ready"

            return self.mutate(change)

    def record_generation(self, allocation_id, report):
        from .generation_publication import bind_parent

        state = self.read()
        library = ScientistModelLibrary(state["definition"]["library_root"])
        stored = library._object("publication_attempts", report["report_id"])
        require(stored == {k: v for k, v in report.items() if k != "report_id"}, "Generation report changed")
        require(stored["kind"] == "model_generation_publication_attempt", "Not a generation publication")
        snapshot = library.goal(safe_child(library.owner, stored["origin_goal"]))
        parent = self.goal(allocation_id)
        bind_parent(snapshot, parent)
        bind_goal(self.specifications()[allocation_id], parent)

        def save(s):
            item = s["allocations"][allocation_id]
            reports = item.setdefault("generation_publications", [])
            if not any(r["report_id"] == report["report_id"] for r in reports):
                reports.append(report)
            item["entries"] = sorted(
                {*item["entries"], *(o["entry_id"] for o in stored["outcomes"] if o["status"] == "published")}
            )

        self.mutate(save)
        self.audit_publications()

    def audit_publications(self):
        state = self.read()
        policy = counting_policy(state["definition"])
        specifications = self.specifications(state)
        by_goal = {str(safe_child(self.root, "goals/" + name)): row for name, row in specifications.items()}
        coverage_records, coverage_cache = [], {}
        library = ScientistModelLibrary(state["definition"]["library_root"])
        own_goals = {str(safe_child(self.root, "goals/" + name)) for name in state["allocations"]}
        recorded = {entry for item in state["allocations"].values() for entry in item["entries"]}
        verified, external, stages = [], [], set()
        for entry in sorted(recorded):
            try:
                value = library.resolve(entry, replay_training=True)
                parent = value["origin"].get("generation_milestone", {}).get("parent_goal", value["origin_goal"])
                if parent in own_goals and parent != value["origin_goal"]:
                    from .generation_publication import bind_parent

                    bind_parent(Path(value["origin_goal"]), Path(parent))
                if parent in own_goals and policy == SCIENTIFIC_SCOPES:
                    allocation = by_goal[parent]
                    bind_goal(allocation, Path(parent))
                    scope = resolve_scope(allocation, cache=coverage_cache)
                    require(value["origin"]["stage"] == scope["stage"], "Published stage differs from bound scope")
                    coverage_records.append({"entry_id": entry, **scope})
            except Exception as exc:
                failure = {"schema": 1, "status": "failed", "entry_id": entry, "error_type": type(exc).__name__}
                path = self.root / "publication_audits" / (digest(failure) + ".json")
                atomic_json(path, failure)
                self.mutate(
                    lambda s, path=path: s.update(
                        status="publication_invalid",
                        verified_entry_ids=[],
                        verified_scope_count=0,
                        last_publication_audit=str(path.relative_to(self.root)),
                    )
                )
                raise
            (verified if parent in own_goals else external).append(entry)
            if parent in own_goals:
                stages.add(value["origin"]["stage"])
        report = {
            "schema": 1,
            "verified_entry_ids": verified,
            "external_duplicate_entry_ids": external,
            "verified_stages": sorted(stages),
            "required_stages": state["definition"]["required_stages"],
            "target_models": state["definition"]["target_models"],
            "target_met": len(verified) >= state["definition"]["target_models"]
            and set(state["definition"]["required_stages"]).issubset(stages),
        }
        if policy == SCIENTIFIC_SCOPES:
            report.update(summarize_scopes(coverage_records))
            report["target_met"] = report["primary_count"] >= report["target_models"] and set(
                report["required_stages"]
            ).issubset(stages)
        if execution_policy(state["definition"]) == ALL_ALLOCATIONS:
            report.update(allocation_execution(state))
            report["completion_met"] = report["target_met"] and report["execution_complete"]
        path = self.root / "publication_audits" / (digest(report) + ".json")
        atomic_json(path, report)

        def save(s):
            s.update(verified_entry_ids=verified, last_publication_audit=str(path.relative_to(self.root)))
            if policy == SCIENTIFIC_SCOPES:
                s["verified_scope_count"] = report["primary_count"]

        self.mutate(save)
        return report


async def run_campaign(store, executor):
    """Executor owns one existing Scientist goal; publication is independently checked here."""
    with store.lease():
        state = store.read()
        require(state["inflight"] is None, "Interrupted reservation requires explicit recovery")
        require(state["status"] in {"ready", "active"}, "Campaign is not runnable")
        owner = process_identity()
        store.mutate(lambda s: s.update(status="active", owner=owner))
        try:
            while True:
                state = store.read()
                if state["status"] != "active":
                    return store.mutate(lambda s: s.update(owner=None))
                specifications = store.specifications(state)
                pending = [name for name in specifications if state["allocations"][name]["status"] == "queued"]
                run_all = execution_policy(state["definition"]) == ALL_ALLOCATIONS
                if not pending or (
                    not run_all and len(state["verified_entry_ids"]) >= state["definition"]["target_models"]
                ):
                    audit = store.audit_publications()
                    completion_met = audit["target_met"] and (not run_all or audit["execution_complete"])
                    if completion_met or not pending:
                        status = "completed" if completion_met else "needs_tasks"
                        return store.mutate(lambda s, status=status: s.update(status=status, owner=None))
                name = pending[0]
                allocation = specifications[name]
                model_tasks.load_task(allocation["task"])
                goal = store.goal(name)
                before = (
                    bind_goal(allocation, goal)["usage"]
                    if (goal / "goal.json").exists()
                    else dict.fromkeys(RESOURCES, 0)
                )
                reservation = {k: max(0, allocation["goal_budget"][k] - before[k]) for k in RESOURCES}
                if any(n <= 0 or state["usage"][k] + n > state["limits"][k] for k, n in reservation.items()):
                    return store.mutate(lambda s: s.update(status="budget_exhausted", owner=None))

                def reserve(s, name=name, reservation=reservation, before=before):
                    s["inflight"] = {
                        "allocation": name,
                        "reservation": reservation,
                        "previous_usage": before,
                        "owner": owner,
                    }
                    s["allocations"][name]["status"] = "running"
                    for k, n in reservation.items():
                        s["usage"][k] += n

                store.mutate(reserve)
                started = time.monotonic()
                result, publication = None, None
                try:
                    result = await executor(allocation, goal)
                    current = bind_goal(allocation, goal)
                    require(result == current, "Executor result differs from saved goal checkpoint")
                    charged = budget({k: current["usage"][k] - before[k] for k in RESOURCES})
                    status = current["status"]
                    if status == "completed":
                        publication = ScientistModelLibrary(state["definition"]["library_root"]).publish_goal(goal)
                        status = "completed" if publication["all_published"] else "publication_rejected"
                    elif status == "waiting_for_input":
                        status = "waiting_for_model"
                    elif status not in {"paused", "budget_exhausted", "interrupted", "cancelled"}:
                        raise ValueError("Goal did not yield a terminal or resumable checkpoint")
                    charged["wall_seconds"] = max(charged["wall_seconds"], time.monotonic() - started)
                    attempt = {
                        "status": status,
                        "usage_known": True,
                        "charged": charged,
                        "goal_sha256": file_hash(goal / "goal.json"),
                        "publication": publication,
                    }
                except Exception as exc:
                    charged = reservation
                    status = "failed"
                    attempt = {
                        "status": status,
                        "usage_known": False,
                        "charged": charged,
                        "error_type": type(exc).__name__,
                    }

                def settle(
                    s,
                    charged=charged,
                    reservation=reservation,
                    name=name,
                    status=status,
                    attempt=attempt,
                    publication=publication,
                ):
                    for k in RESOURCES:
                        s["usage"][k] += charged[k] - reservation[k]
                    item = s["allocations"][name]
                    item["status"] = status
                    item["attempts"].append(attempt)
                    if publication:
                        item["entries"] = sorted(
                            {
                                *item["entries"],
                                *(x["entry_id"] for x in publication["outcomes"] if x["status"] == "published"),
                            }
                        )
                    s["inflight"] = None
                    if s["status"] != "paused" and status in {"waiting_for_model", "paused"}:
                        s["status"] = status
                    if any(s["usage"][k] > s["limits"][k] for k in RESOURCES):
                        s["status"] = "budget_exhausted"

                store.mutate(settle)
                if publication:
                    store.audit_publications()
        finally:
            store.mutate(lambda s: s.update(owner=None), verify_source=False)


def bind_goal(allocation, root):
    state = GoalStore(root).read()
    definition = state["definition"]
    require(
        definition["condition"] == "science_goal"
        and Profile.from_dict(definition["profile"]) == Profile.from_dict(allocation["profile"]),
        "Child goal differs from admitted Scientist profile",
    )
    require(definition.get("admission") == model_tasks.admission(allocation["task"]), "Child task admission changed")
    require(all(definition["goal"][k] == v for k, v in allocation["goal_budget"].items()), "Child goal budget changed")
    require(read(root / "model_task.json") == allocation["task"], "Child task definition changed")
    protocol = read(root / "protocol.json")
    memory_policy = training_attempt_memory.verify_goal(state, protocol)
    if training_attempt_memory.FIELD in allocation:
        require(
            memory_policy == allocation[training_attempt_memory.FIELD],
            "Child training-attempt memory allocation changed",
        )
    else:
        require(
            memory_policy != training_attempt_memory.CUMULATIVE_POLICY,
            "Cumulative memory requires an explicit campaign allocation",
        )
    feedback = context_feedback.verify_goal(state, protocol)
    require(
        feedback == context_feedback.policy_for(allocation, plugin_id=allocation["task"]["plugin_id"]),
        "Child baseline-feedback allocation changed",
    )
    require(
        protocol.get("evolution", {}).get("generations", 1) == allocation["generations"],
        "Child lineage allocation changed",
    )
    frozen = library_experience.read_frozen(root, state, protocol)
    if "experience" not in allocation:
        require(frozen is None, "Unexpected campaign library experience")
    else:
        policy = library_experience.campaign_policy(allocation["experience"])
        require(frozen is not None, "Missing campaign library experience")
        request = frozen["manifest"]["selection"]
        campaign = ModelCampaignStore(root.parent.parent)
        campaign_state = campaign.read()
        require(
            request["condition"] == policy["condition"]
            and request["library_root"] == campaign_state["definition"]["library_root"],
            "Campaign experience condition or owner changed",
        )
        require(
            request.get("overlap_policy") == policy.get("overlap_policy"),
            "Campaign source overlap policy changed",
        )
        require(
            request.get("adaptation_policy") == policy.get("adaptation_policy"),
            "Campaign source adaptation policy changed",
        )
        if policy["selection"] == "explicit":
            require(request["entry_ids"] == policy["entry_ids"], "Campaign donor selection changed")
        else:
            prior = prior_publications(campaign, allocation["id"])
            require(
                len(request["entry_ids"]) <= policy["max_entries"] and set(request["entry_ids"]) <= set(prior),
                "Campaign experience contains a later or unadmitted publication",
            )
    criteria = {"accepted_model", "fresh_session_reuse"} | (
        {"verified_lineage"} if allocation["generations"] > 1 else set()
    )
    require(set(definition["goal"]["criteria"]) == criteria, "Child goal completion criteria changed")
    return state


def prior_publications(store, allocation_id):
    """Earlier allocations only, newest first; deterministic within allocation."""
    state = store.read()
    earlier = []
    for name in store.specifications(state):
        if name == allocation_id:
            break
        earlier.append(name)
    else:
        raise ValueError("Unknown campaign allocation")
    return list(
        dict.fromkeys(
            entry
            for name in reversed(earlier)
            for entry in sorted(state["allocations"][name]["entries"])
            if entry in state["verified_entry_ids"]
        )
    )
