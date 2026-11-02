"""Persistent acquisition across shared Scientist goals, counting scientific scopes."""

from __future__ import annotations

import copy
import math
import re
import time
from pathlib import Path

from agent.harness.contracts import Profile, digest, file_hash
from agent.harness.native_expert_team import NATIVE_EXPERT_HARNESSES
from agent.science_goal import GoalSpec, GoalStore

from . import context_feedback_dispatch as feedback
from . import model_tasks, shared_campaign_experience, shared_library_experience
from .research_project_catalog import scope_key
from .scientist_model_campaign import (
    RESOURCES,
    ModelCampaignStore,
    budget,
    process_identity,
    process_live,
    read,
    require,
)
from .scientist_model_library import ScientistModelLibrary
from .shared_goal_history import SUPPORTED_POLICIES
from .shared_model_bundle import admit
from .shared_model_publication import validate_shared_goal
from .storage import atomic_json, safe_child

SCHEMA = "shared_model_campaign.v1"
CRITERIA = ("all_targets_accepted", "all_targets_freshly_reused")
TERMINAL = {"completed", "budget_exhausted", "failed", "interrupted", "cancelled"}


def task_scope(task):
    scope = task["scope"]
    return scope_key(
        {
            "protein_scope": {"kind": "protein", "identifier": scope["protein"]},
            "property": scope["target"],
            "scenario": scope.get("context", {"assay": scope["assay"]}),
        },
        task,
    )


def admit_batches(batches, *, batch_ids=(), scope_ids=()):
    require(isinstance(batches, list) and 1 <= len(batches) <= 10000, "Declare 1..10000 shared batches")
    seen, scopes, result = set(batch_ids), set(scope_ids), []
    for item in batches:
        require(
            set(item) - {shared_campaign_experience.FIELD}
            == {"id", "tasks", "profile", "goal_budget", "training_baseline_feedback", "shared_training_history"},
            "Invalid shared batch fields",
        )
        name = item["id"]
        require(
            isinstance(name, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,79}", name) is not None and name not in seen,
            "Duplicate or unsafe shared batch identity",
        )
        seen.add(name)
        admit(item["tasks"])
        if shared_campaign_experience.FIELD in item:
            shared_campaign_experience.policy(item[shared_campaign_experience.FIELD], item["tasks"])
        profile = Profile.from_dict(item["profile"])
        require(
            profile.mode == "science_expert" and profile.harness in NATIVE_EXPERT_HARNESSES,
            "Native audited Scientist profile required",
        )
        spec = GoalSpec("Shared campaign allocation", CRITERIA, **item["goal_budget"])
        require(spec.max_cycles >= 2, "Allow development and fresh reuse")
        require(item["shared_training_history"] in SUPPORTED_POLICIES, "Unsupported shared history")
        for task in item["tasks"]:
            feedback.policy_for(
                {feedback.FIELD: item["training_baseline_feedback"], "scope": task["scope"]},
                plugin_id=task["plugin_id"],
            )
            identity = task_scope(task)
            require(identity not in scopes, "Scientific scope already allocated or explicitly excluded")
            scopes.add(identity)
        result.append(copy.deepcopy(item))
    return result


class SharedCampaignStore(ModelCampaignStore):
    """Reuse Linux leases/atomic mutation, with a separate immutable plan schema."""

    @classmethod
    def create(
        cls,
        root,
        *,
        objective,
        target_scopes,
        library_root,
        repository,
        implementation,
        batches,
        resources,
        excluded_scopes=(),
        required_stages=(),
    ):
        require(isinstance(objective, str) and objective.strip(), "Explicit research objective required")
        require(type(target_scopes) is int and target_scopes > 0, "Positive scientific-scope target required")
        require(
            len(set(excluded_scopes)) == len(excluded_scopes)
            and all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) for x in excluded_scopes),
            "Invalid excluded scopes",
        )
        require(
            set(required_stages) <= set(model_tasks.STAGES) and len(set(required_stages)) == len(required_stages),
            "Invalid required stages",
        )
        admitted = admit_batches(batches, scope_ids=excluded_scopes)
        budget(resources)
        store = cls(root)
        library = ScientistModelLibrary(library_root)
        library.goal(store.root)
        repository = Path(repository).resolve()
        require(bool(implementation), "Frozen implementation required")
        require(
            all(file_hash(safe_child(repository, name)) == expected for name, expected in implementation.items()),
            "Source changed before campaign creation",
        )
        definition = {
            "schema": SCHEMA,
            "objective": objective,
            "target_scopes": target_scopes,
            "library_root": str(library.root),
            "repository": str(repository),
            "implementation": dict(implementation),
            "excluded_scopes": sorted(excluded_scopes),
            "required_stages": list(required_stages),
        }
        store.root.mkdir(parents=True, exist_ok=False)
        state = {
            "schema": SCHEMA,
            "definition": definition,
            "definition_sha256": digest(definition),
            "plans": [],
            "batches": {},
            "status": "ready",
            "inflight": None,
            "owner": None,
            "limits": dict.fromkeys(RESOURCES, 0),
            "usage": dict.fromkeys(RESOURCES, 0),
            "events": [],
            "last_audit": None,
        }
        atomic_json(store.root / "campaign.json", state)
        store.extend(admitted, resources=resources)
        return store

    def specifications(self, state=None):
        state = state or self.read()
        return {
            batch["id"]: batch
            for identity in state["plans"]
            for batch in read(safe_child(self.root, "plans/" + identity + ".json"))["batches"]
        }

    def read(self, *, verify_source=True):
        state = read(self.root / "campaign.json")
        require(
            state["schema"] == SCHEMA and digest(state["definition"]) == state["definition_sha256"],
            "Shared campaign definition changed",
        )
        previous, names, scopes, limits = (
            None,
            set(),
            set(state["definition"]["excluded_scopes"]),
            dict.fromkeys(RESOURCES, 0),
        )
        for identity in state["plans"]:
            require(
                isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity), "Invalid shared plan identity"
            )
            plan = read(safe_child(self.root, "plans/" + identity + ".json"))
            require(digest(plan) == identity and plan["previous"] == previous, "Shared plan chain changed")
            for batch in plan["batches"]:
                require(batch["id"] not in names, "Repeated shared batch")
                names.add(batch["id"])
                for task in batch["tasks"]:
                    scope = task_scope(task)
                    require(scope not in scopes, "Repeated scientific scope")
                    scopes.add(scope)
            for key, value in budget(plan["resources"]).items():
                limits[key] += value
            previous = identity
        require(set(state["batches"]) == names and state["limits"] == limits, "Shared campaign inventory changed")
        budget(state["usage"])
        recorded = dict.fromkeys(RESOURCES, 0)
        for item in state["batches"].values():
            for attempt in item["attempts"]:
                for key, amount in budget(attempt["charged"]).items():
                    recorded[key] += amount
        if state["inflight"] is not None:
            require(state["inflight"]["batch"] in state["batches"], "Unknown reserved batch")
            for key, amount in budget(state["inflight"]["reservation"]).items():
                recorded[key] += amount
        require(
            all(
                math.isclose(recorded[key], state["usage"][key], rel_tol=1e-12, abs_tol=1e-9)
                if key == "wall_seconds"
                else recorded[key] == state["usage"][key]
                for key in RESOURCES
            ),
            "Campaign usage differs from settled attempts and active reservation",
        )
        if verify_source:
            repository = Path(state["definition"]["repository"])
            require(
                all(
                    file_hash(safe_child(repository, name)) == expected
                    for name, expected in state["definition"]["implementation"].items()
                ),
                "Frozen campaign implementation changed",
            )
        return state

    def extend(self, batches, *, resources):
        with self.lease():
            state = self.read()
            require(
                state["inflight"] is None and state["status"] in {"ready", "paused", "needs_tasks", "budget_exhausted"},
                "Extend a quiescent shared campaign",
            )
            specs = self.specifications(state)
            scopes = set(state["definition"]["excluded_scopes"]) | {
                task_scope(task) for item in specs.values() for task in item["tasks"]
            }
            batches = admit_batches(batches, batch_ids=specs, scope_ids=scopes)
            plan = {
                "schema": SCHEMA,
                "previous": state["plans"][-1] if state["plans"] else None,
                "batches": batches,
                "resources": budget(resources),
            }
            identity = digest(plan)
            path = safe_child(self.root, "plans/" + identity + ".json")
            require(not path.exists(), "Shared plan already exists")
            atomic_json(path, plan)
            path.chmod(0o444)

            def save(value):
                value["plans"].append(identity)
                for item in batches:
                    value["batches"][item["id"]] = {"status": "queued", "attempts": []}
                for key, amount in resources.items():
                    value["limits"][key] += amount
                value["events"].append({"event": "plan_added", "plan": identity})
                if value["status"] != "paused":
                    value["status"] = "ready"

            return self.mutate(save)

    def goal(self, identity):
        require(identity in self.read()["batches"], "Unknown shared batch")
        return safe_child(self.root, "goals/" + identity)

    def pause(self):
        state = self.mutate(lambda s: s.update(status="paused"), verify_source=False)
        if state["inflight"]:
            path = safe_child(self.root, "goals/" + state["inflight"]["batch"])
            if (path / "goal.json").is_file():
                GoalStore(path).pause()
        return state

    def resume(self):
        with self.lease():
            state = self.read()
            require(
                state["status"] in {"paused", "waiting_for_model"} and state["inflight"] is None,
                "Resume requires a quiescent paused campaign",
            )
            for name, item in state["batches"].items():
                if item["status"] in {"paused", "waiting_for_model"}:
                    child = GoalStore(self.goal(name))
                    require(child.read()["status"] in {"paused", "waiting_for_input"}, "Child is not resumable")
                    child.resume()

            def save(value):
                for item in value["batches"].values():
                    if item["status"] in {"paused", "waiting_for_model"}:
                        item["status"] = "queued"
                value["status"] = "ready"

            return self.mutate(save)

    def recover(self):
        """Charge an unknown interrupted reservation; never rerun that batch."""
        with self.lease():

            def save(state):
                inflight = state["inflight"]
                require(
                    inflight is not None and not process_live(inflight["owner"]),
                    "No dead shared reservation to recover",
                )
                item = state["batches"][inflight["batch"]]
                item["status"] = "interrupted"
                item["attempts"].append(
                    {
                        "status": "interrupted",
                        "usage_known": False,
                        "charged": inflight["reservation"],
                        "publication": None,
                    }
                )
                state.update(inflight=None, owner=None, status="ready")

            return self.mutate(save)

    def audit_publications(self):
        state = self.read()
        specs = self.specifications(state)
        library = ScientistModelLibrary(state["definition"]["library_root"])
        covered, stages, batches = set(), set(), []
        for name, item in state["batches"].items():
            publications = [attempt["publication"] for attempt in item["attempts"] if attempt.get("publication")]
            if not publications:
                continue
            require(len(publications) == 1, "Shared batch publication repeated")
            report = publications[0]
            stored = library._object("publication_attempts", report["report_id"])
            require(
                stored == {key: value for key, value in report.items() if key != "report_id"},
                "Shared publication report changed",
            )
            goal = self.goal(name)
            bind_goal(self, specs[name], goal)
            if shared_campaign_experience.FIELD in specs[name]:
                shared_campaign_experience.verify_selection(self, specs[name], goal=goal, replay=True)
            require(
                stored["kind"] == "shared_model_goal_publication_attempt"
                and stored["origin_goal"] == str(goal.relative_to(library.owner)),
                "Foreign shared publication",
            )
            verified = {row["origin"]["task_id"]: row for row in validate_shared_goal(goal, replay_training=True)}
            require(
                set(verified) == {outcome["task_id"] for outcome in stored["outcomes"]},
                "Shared publication target coverage changed",
            )
            tasks = {task["id"]: task for task in specs[name]["tasks"]}
            for outcome in stored["outcomes"]:
                value = verified[outcome["task_id"]]
                entry = library._object("entries", outcome["entry_id"])
                require(
                    outcome["status"] == "published"
                    and outcome["model_id"] == value["origin"]["model_id"]
                    and entry["key"] == value["key"]
                    and entry["packet_id"] == digest(value["packet"]),
                    "Published model binding changed",
                )
                task = tasks[outcome["task_id"]]
                identity = task_scope(task)
                require(identity not in covered, "Repeated scientific scope in publication")
                covered.add(identity)
                stages.add(task["stage"])
            batches.append({"id": name, "verified_scopes": len(verified), "goal_sha256": file_hash(goal / "goal.json")})
        terminal = all(item["status"] in TERMINAL for item in state["batches"].values()) and state["inflight"] is None
        report = {
            "schema": SCHEMA,
            "definition_sha256": state["definition_sha256"],
            "plans": state["plans"],
            "verified_scope_ids": sorted(covered),
            "verified_scope_count": len(covered),
            "verified_stages": sorted(stages),
            "batches": batches,
            "all_batches_terminal": terminal,
            "target_met": len(covered) >= state["definition"]["target_scopes"]
            and set(state["definition"]["required_stages"]) <= stages,
        }
        identity = digest(report)
        atomic_json(self.root / "audits" / (identity + ".json"), report)
        self.mutate(lambda s: s.update(last_audit=identity))
        return report


def bind_goal(store, batch, root):
    state = GoalStore(root).read()
    definition = state["definition"]
    campaign = store.read()["definition"]
    require(
        definition["condition"] == "science_goal"
        and Profile.from_dict(definition["profile"]) == Profile.from_dict(batch["profile"]),
        "Shared child profile changed",
    )
    require(
        definition["goal"] == {"objective": campaign["objective"], "criteria": list(CRITERIA), **batch["goal_budget"]},
        "Shared child goal budget or criteria changed",
    )
    protocol = read(root / "protocol.json")
    expected = {
        "schema": "shared_scientist_goal.v1",
        "tasks": [model_tasks.admission(task) for task in batch["tasks"]],
        "training_baseline_feedback": batch["training_baseline_feedback"],
        "implementation_sha256": digest(campaign["implementation"]),
        "shared_training_history": batch["shared_training_history"],
    }
    if shared_campaign_experience.FIELD in batch:
        selection = shared_campaign_experience.verify_selection(store, batch, goal=root)
        expected[shared_library_experience.FIELD] = {"snapshot_sha256": selection["screen"]["selected_snapshot_sha256"]}
    require(
        protocol["schema"] == "shared_scientist_goal.v1"
        and protocol["admission"] == definition["admission"] == expected,
        "Shared child allocation changed",
    )
    require(protocol["implementation"] == campaign["implementation"], "Shared child implementation changed")
    require(
        protocol["target_training_baseline_feedback"] == batch["training_baseline_feedback"]
        and protocol["training_attempt_memory"] == "none",
        "Shared child feedback policy changed",
    )
    require(
        [row["id"] for row in protocol["targets"]] == [task["id"] for task in batch["tasks"]],
        "Shared target order changed",
    )
    for task, binding in zip(batch["tasks"], protocol["targets"], strict=True):
        require(
            read(safe_child(root, "targets/" + task["id"] + "/model_task.json")) == task
            and binding["task_sha256"] == digest(task),
            "Shared child task changed",
        )
    return state


async def run_campaign(store, executor):
    with store.lease():
        state = store.read()
        require(
            state["inflight"] is None and state["status"] in {"ready", "active"},
            "Shared campaign is not runnable; recover or resume explicitly",
        )
        owner = process_identity()
        store.mutate(lambda s: s.update(status="active", owner=owner))
        try:
            while True:
                state = store.read()
                if state["status"] != "active":
                    return store.mutate(lambda s: s.update(owner=None))
                specs = store.specifications(state)
                pending = [name for name in specs if state["batches"][name]["status"] == "queued"]
                if not pending:
                    try:
                        audit = store.audit_publications()
                    except Exception:
                        store.mutate(lambda s: s.update(status="publication_invalid"))
                        raise
                    status = "completed" if audit["target_met"] and audit["all_batches_terminal"] else "needs_tasks"
                    return store.mutate(lambda s, status=status: s.update(status=status, owner=None))
                name = pending[0]
                batch, goal = specs[name], store.goal(name)
                admit(batch["tasks"])
                before = (
                    bind_goal(store, batch, goal)["usage"]
                    if (goal / "goal.json").exists()
                    else dict.fromkeys(RESOURCES, 0)
                )
                reservation = {key: max(0, batch["goal_budget"][key] - before[key]) for key in RESOURCES}
                if any(
                    value <= 0 or state["usage"][key] + value > state["limits"][key]
                    for key, value in reservation.items()
                ):
                    return store.mutate(lambda s: s.update(status="budget_exhausted", owner=None))

                def reserve(value, name=name, reservation=reservation, before=before):
                    value["inflight"] = {
                        "batch": name,
                        "reservation": reservation,
                        "previous_usage": before,
                        "owner": owner,
                    }
                    value["batches"][name]["status"] = "running"
                    for key, amount in reservation.items():
                        value["usage"][key] += amount

                store.mutate(reserve)
                started, publication = time.monotonic(), None
                try:
                    returned = await executor(batch, goal)
                    current = bind_goal(store, batch, goal)
                    require(
                        returned == current and current["inflight"] is None,
                        "Shared executor checkpoint differs or is incomplete",
                    )
                    charged = budget({key: current["usage"][key] - before[key] for key in RESOURCES})
                    require(
                        all(charged[key] <= reservation[key] for key in ("model_calls", "tool_calls")),
                        "Shared child exceeded reserved requests or tools",
                    )
                    status = current["status"]
                    require(
                        status
                        in {"completed", "budget_exhausted", "paused", "waiting_for_input", "interrupted", "cancelled"},
                        "Shared child has no terminal or resumable checkpoint",
                    )
                    if status in {"completed", "budget_exhausted"} and any(
                        "reuse_receipt" in row for row in current["progress"].get("targets", {}).values()
                    ):
                        publication = ScientistModelLibrary(state["definition"]["library_root"]).publish_goal(goal)
                        require(publication["all_published"], "Shared publication failed")
                    if status == "waiting_for_input":
                        status = "waiting_for_model"
                    charged["wall_seconds"] = max(charged["wall_seconds"], time.monotonic() - started)
                    attempt = {
                        "status": status,
                        "usage_known": True,
                        "charged": charged,
                        "goal_sha256": file_hash(goal / "goal.json"),
                        "publication": publication,
                    }
                except Exception as exc:
                    status, charged, publication = "failed", reservation, None
                    attempt = {
                        "status": status,
                        "usage_known": False,
                        "charged": charged,
                        "publication": None,
                        "error_type": type(exc).__name__,
                    }

                def settle(value, charged=charged, reservation=reservation, name=name, status=status, attempt=attempt):
                    for key in RESOURCES:
                        value["usage"][key] += charged[key] - reservation[key]
                    value["batches"][name]["status"] = status
                    value["batches"][name]["attempts"].append(attempt)
                    value["inflight"] = None
                    if value["status"] != "paused" and status in {"paused", "waiting_for_model"}:
                        value["status"] = status
                    if any(value["usage"][key] > value["limits"][key] for key in RESOURCES):
                        value["status"] = "budget_exhausted"

                store.mutate(settle)
        finally:
            store.mutate(lambda s: s.update(owner=None), verify_source=False)
