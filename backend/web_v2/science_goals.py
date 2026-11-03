"""Session-owned persistent Science goals with detached workers.

Drivers declare scientific tasks and verified criteria. The supervisor owns
process identity and API lifecycle, independently of model/harness selection.
"""

from __future__ import annotations

from backend.web_v2.research_delivery import project_summary

import fcntl
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from agent.harness import Profile, builtin_registry
from agent.harness.catalog import ExecutionCatalog, resolve_credential
from agent.harness.provider_config import online as provider_online
from agent.harness.contracts import digest
from agent.rsi.campaigns.case_plugins import PLUGINS
from agent.rsi.campaigns.paper_campaign import SUITES, admitted_campaign, campaign_cases
from agent.rsi.engine.reproduction_goal_contract import CASES, CRITERIA, allocation
from agent.rsi.engine.storage import atomic_json, file_lock
from agent.rsi.engine.versions import VersionStore
from agent.science_goal import GoalStore
from backend.web_v2.paper_step_goals import available_steps, public_step
from backend.web_v2.paper_step_goals import drivers as paper_step_drivers

ROOT = Path(__file__).resolve().parents[2]
NONTERMINAL = {"preparing", "active", "paused", "interrupted", "waiting_for_input"}


class GoalConflict(ValueError):
    pass


@dataclass(frozen=True)
class GeneratedModelDriver:
    id: str = "generated_model.v1"
    title: str = "Protein model development and verified reuse"
    criteria: tuple[str, ...] = ("accepted_model", "fresh_session_reuse")
    cases: tuple[str, ...] = ("alde_parpgb", "mcba_itopride")

    def prepare_command(self, service, folder, spec):
        params = spec["parameters"]
        if set(params) != {"case"} or params["case"] not in self.cases:
            raise ValueError("Select a registered protein modelling case")
        command = [
            str(service.python),
            str(service.root / "scripts/rsi/run_science_goal.py"),
            "create",
            "--output",
            str(folder),
            "--profile",
            str(service.prepared_profile(folder)),
            "--objective",
            spec["objective"],
            "--case",
            params["case"],
        ]
        for key, value in spec["budget"].items():
            command += ["--" + key.replace("_", "-"), str(value)]
        return command

    def run_command(self, service, folder, action):
        command = [
            str(service.python),
            "-u",
            str(service.root / "scripts/rsi/run_science_goal.py"),
            action,
            "--output",
            str(folder),
        ]
        profile = service.frozen_profile(folder)
        bridge = service.frozen_credential_path(folder)
        if not os.environ.get(profile.model.credential_env) and bridge:
            command += ["--credential-toml", str(bridge)]
        return command


@dataclass(frozen=True)
class AutonomousResearchDriver(GeneratedModelDriver):
    id: str = "autonomous_research.v1"
    title: str = "Autonomous research"
    cases: tuple[str, ...] = ()
    criteria: tuple[str, ...] = ("research_plan_executed", "research_outputs_reviewed", "capability_admission_resolved")
    maximum_cycles: int = 32
    maximum_model_calls: int = 768
    maximum_tool_calls: int = 384
    maximum_wall_seconds: int = 28 * 86400

    def prepare_command(self, service, folder, spec):
        if spec["parameters"]:
            raise ValueError("Autonomous research does not require a template or case")
        command = [str(service.python), str(service.root / "scripts/rsi/run_autonomous_research.py"), "create", "--output", str(folder), "--profile", str(service.prepared_profile(folder)), "--objective", spec["objective"]]
        for key, value in spec["budget"].items():
            command += ["--" + key.replace("_", "-"), str(value)]
        return command

    def run_command(self, service, folder, action):
        command = super().run_command(service, folder, action)
        command[2] = str(service.root / "scripts/rsi/run_autonomous_research.py")
        return command


@dataclass(frozen=True)
class EvolvingModelDriver(GeneratedModelDriver):
    id: str = "generated_model_evolution.v1"
    title: str = "Iterative model development and capability registration"
    cases: tuple[str, ...] = ("alde_parpgb", "mcba_itopride", "nucb", "gfp")
    criteria: tuple[str, ...] = ("accepted_model", "fresh_session_reuse", "verified_lineage")

    def prepare_command(self, service, folder, spec):
        params = spec["parameters"]
        if set(params) != {"case", "generations"} or params["generations"] not in {"2", "3"}:
            raise ValueError("Choose two or three verified generations")
        generations = int(params["generations"])
        if spec["budget"]["max_cycles"] < generations * 2:
            raise ValueError("Allow at least two episodes per requested generation")
        base = {**spec, "parameters": {"case": params["case"]}}
        command = super().prepare_command(service, folder, base)
        script = "run_classifier_evolution.py" if params["case"] in {"nucb", "gfp"} else "run_science_evolution.py"
        command[1] = str(service.root / "scripts/rsi" / script)
        return command + ["--generations", str(generations)]

    def run_command(self, service, folder, action):
        command = super().run_command(service, folder, action)
        case = json.loads((folder / "protocol.json").read_text()).get("case")
        script = "run_classifier_evolution.py" if case in {"nucb", "gfp"} else "run_science_evolution.py"
        command[2] = str(service.root / "scripts/rsi" / script)
        return command


@dataclass(frozen=True)
class ReproductionDriver(GeneratedModelDriver):
    id: str = "reproduction_goal.v1"
    title: str = "Protein study reproduction and verified workflow reuse"
    criteria: tuple[str, ...] = CRITERIA
    cases: tuple[str, ...] = tuple(CASES)
    maximum_wall_seconds: int = 7 * 86400

    def prepare_command(self, service, folder, spec):
        if spec["budget"]["max_cycles"] < 2:
            raise ValueError("Reproduction and fresh-session reuse require at least two episodes")
        command = super().prepare_command(service, folder, spec)
        command[1] = str(service.root / "scripts/rsi/run_reproduction_goal.py")
        return command

    def run_command(self, service, folder, action):
        command = super().run_command(service, folder, action)
        command[2] = str(service.root / "scripts/rsi/run_reproduction_goal.py")
        return command

    def case_details(self, profile):
        inference = profile.budget.wall_seconds if profile else 900
        result = {}
        for case in self.cases:
            declared = allocation(case)
            result[case] = {
                "scope": PLUGINS[declared["plugin"]].scope,
                "suggested_budget": {
                    "max_cycles": 4,
                    "model_calls": 96,
                    "tool_calls": 48,
                    "wall_seconds": min(self.maximum_wall_seconds, int(2 * (declared["job_timeout"] + inference))),
                },
            }
        return result


@dataclass(frozen=True)
class PaperCampaignDriver(GeneratedModelDriver):
    id: str = "paper_campaign.v1"
    title: str = "Multi-paper reproduction and capability accumulation"
    criteria: tuple[str, ...] = CRITERIA
    cases: tuple[str, ...] = tuple(SUITES)
    maximum_wall_seconds: int = 28 * 86400
    maximum_cycles: int = 32
    maximum_model_calls: int = 768
    maximum_tool_calls: int = 384

    def prepare_command(self, service, folder, spec):
        command = super().prepare_command(service, folder, spec)
        papers = campaign_cases(spec["parameters"]["case"])
        if spec["budget"]["max_cycles"] < 2 * len(papers):
            raise ValueError("Allow at least two episodes per requested paper scope")
        command[1] = str(service.root / "scripts/rsi/run_paper_campaign.py")
        command += ["--capability-library", str(folder.parent / ".paper_capabilities")]
        return command

    def run_command(self, service, folder, action):
        command = super().run_command(service, folder, action)
        command[2] = str(service.root / "scripts/rsi/run_paper_campaign.py")
        return command

    def case_details(self, profile):
        inference = profile.budget.wall_seconds if profile else 900
        result = {}
        for suite in self.cases:
            papers = campaign_cases(suite)
            result[suite] = {
                "scope": "; ".join(p["paper_id"] + ": " + ", ".join(p["covered_steps"]) for p in papers)
                + ". Partial computational reproductions; excluded steps remain unverified.",
                "paper_count": len(papers),
                "minimum_cycles": 2 * len(papers),
                "papers": papers,
                "suggested_budget": {
                    "max_cycles": 4 * len(papers),
                    "model_calls": 96 * len(papers),
                    "tool_calls": 48 * len(papers),
                    "wall_seconds": min(
                        self.maximum_wall_seconds,
                        int(2 * sum(p["allocation"]["job_timeout"] + inference for p in papers)),
                    ),
                },
            }
        return result


def public_paper_progress(folder, state):
    try:
        protocol = json.loads((folder / "protocol.json").read_text())
        if protocol.get("kind") != "paper_campaign.v1":
            return []
        papers = admitted_campaign(protocol, state)
        assessments = (state.get("assessment") or {}).get("papers", {})
        result = []
        for paper in papers:
            checks = assessments.get(paper["case"], {}).get("criteria", {})
            result.append(
                {
                    **{
                        k: paper[k]
                        for k in (
                            "case",
                            "paper_id",
                            "title",
                            "doi",
                            "reproduction_level",
                            "covered_steps",
                            "excluded_steps",
                        )
                    },
                    "verified": all(checks.get(k, {}).get("passed") is True for k in CRITERIA),
                    "full_paper_reproduced": False,
                    "criteria": [{"name": k, "passed": checks.get(k, {}).get("passed", False)} for k in CRITERIA],
                }
            )
        return result
    except (OSError, ValueError, KeyError, TypeError):
        return []


def driver_limits(driver):
    return {
        "max_cycles": getattr(driver, "maximum_cycles", 8),
        "model_calls": getattr(driver, "maximum_model_calls", 192),
        "tool_calls": getattr(driver, "maximum_tool_calls", 96),
        "wall_seconds": getattr(driver, "maximum_wall_seconds", 7200),
    }


def public_reproduction_recipes(folder, state):
    if state["progress"].get("papers") is not None:
        rows = []
        assessments = (state.get("assessment") or {}).get("papers", {})
        for case, progress in state["progress"]["papers"].items():
            rows.extend(
                {**r, "case": case}
                for r in public_reproduction_recipes(
                    folder, {"progress": progress, "assessment": assessments.get(case)}
                )
            )
        return rows
    version = state["progress"].get("recipe_version")
    if not version:
        return []
    result = {"version_id": str(version), "available": False, "reused": False}
    try:
        if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{64}", version):
            raise ValueError("Invalid recipe identity")
        root = folder / "recipes"
        if not root.is_dir() or root.is_symlink():
            raise ValueError("Recipe registry unavailable")
        VersionStore(root).verify(version)
        path = root / "versions" / version / "recipe.json"
        if path.is_symlink():
            raise ValueError("Recipe symlink rejected")
        recipe = json.loads(path.read_text())
        if recipe.get("kind") != "verified_reproduction_recipe":
            raise ValueError("Unexpected capability kind")
        checks = (state.get("assessment") or {}).get("criteria", {})
        schedule = None
        if (folder/'queue.json').is_file():
            from backend.web_v2.job_queue import JobQueue
            schedule = JobQueue().public(json.loads((folder/'queue.json').read_text())['id'])
        result.update(
            available=True,
            plugin=recipe["plugin"],
            scope=recipe["scope"],
            imported=bool(recipe.get("library_origin")),
            reused=bool(state["progress"].get("reuse_receipt") and checks.get("fresh_session_reuse", {}).get("passed")),
        )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return [result]


def goal_activity(folder, state):
    """Bounded user-visible role reports, never private model requests or credentials."""
    activities = []
    indices = [c["index"] for c in state["cycles"]]
    if state["inflight"]:
        indices.append(state["inflight"]["index"])
    roles = {
        "principal_investigator": "PI",
        "computational_biologist": "CB",
        "machine_learning_specialist": "MLS",
        "scientific_critic": "SC",
    }
    for index in indices[-8:]:
        episode = folder / "episodes" / f"cycle_{index:03d}"
        activity_path = episode / "activity.json"
        if activity_path.is_file() and not activity_path.is_symlink() and activity_path.stat().st_size < 1000000:
            for n, row in enumerate(json.loads(activity_path.read_text())):
                if row.get("role") in {"PI", "CB", "MLS", "SC", "REVIEW"}:
                    activities.append({"id": f"{index}-research-{n}", "episode": index + 1, "role": row["role"], "timestamp": row.get("timestamp"), "content": str(row.get("content", ""))[:6000]})
        paths = [
            ("model", "execution/graph_history.json"),
            ("step", "runtime/graph_history.json"),
            ("launch", "reproduction/launch/graph_history.json"),
            ("review", "reproduction/review/graph_history.json"),
        ]
        for stage, relative in paths:
            path = episode / relative
            try:
                path.resolve().relative_to(folder.resolve())
                if path.is_symlink() or path.stat().st_size > 2_000_000:
                    continue
                rows = json.loads(path.read_text())
                if not isinstance(rows, list):
                    continue
            except (OSError, ValueError):
                continue
            for n, row in enumerate(rows[-60:]):
                if not isinstance(row, dict):
                    continue
                role = roles.get(row.get("role_id"))
                if role and isinstance(row.get("content"), str):
                    activity = {
                        "id": f"{index}-{n}" if stage == "model" else f"{index}-{stage}-{n}",
                        "episode": index + 1,
                        "role": role,
                        "content": row["content"][:6000],
                    }
                    if stage != "model":
                        activity["stage"] = stage
                    activities.append(activity)
    return activities[-80:]


def research_live(folder, state):
    if not state["cycles"] and not state["inflight"]:
        return {}
    index = state["inflight"]["index"] if state["inflight"] else state["cycles"][-1]["index"]
    episode = folder / "episodes" / f"cycle_{index:03d}"
    result = {}
    for key, relative in [("tool", "current_tool.json"), ("execution", "tools/execution.json"), ("phase", "phase.json"), ("checkpoint", "tools/checkpoint.json"), ("stages", "stages.json"), ("timeline", "timeline.json")]:
        path = episode / relative
        if path.is_file() and not path.is_symlink() and path.stat().st_size < 100000:
            try: result[key] = json.loads(path.read_text())
            except ValueError: pass
    workspace = folder / "workspace"
    if workspace.is_dir():
        result["files"] = [{"path": str(p.relative_to(workspace)), "bytes": p.stat().st_size} for p in sorted(workspace.rglob("*")) if p.is_file() and not p.is_symlink() and not any(part.startswith(".") for part in p.relative_to(workspace).parts)][:200]
    from agent.harness.session_boundary import public_value
    return public_value(result, workspace)


def process_identity(pid):
    """Linux PID plus boot/start identity prevents stale-PID reuse."""
    try:
        content = (Path("/proc") / str(pid) / "stat").read_text()
        fields = content[content.rfind(")") + 2 :].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": int(pid),
            "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (OSError, ValueError, IndexError):
        return None


class ScienceGoalService:
    def __init__(
        self,
        root=ROOT,
        *,
        state_root=None,
        profile_path=None,
        credential_path=None,
        python=None,
        drivers=None,
        catalog_path=None,
    ):
        self.root = Path(root).resolve()
        self.state_root = Path(state_root or (os.getenv("VENUS_SCIENCE_GOAL_ROOT") if self.root == ROOT else None) or self.root / "research_runs/rsi/web_goals").resolve()
        self.profile_path = Path(profile_path or self.root / "configs/harness/free_science_goal.json").resolve()
        self.credential_path = Path(
            credential_path or self.root / "research_runs/rsi/runtime_configs/glm51/config.toml"
        ).resolve()
        configured_catalog = catalog_path or os.environ.get("VENUS_EXECUTION_CATALOG")
        candidate = (
            Path(configured_catalog)
            if configured_catalog
            else self.root / "configs/harness/free_execution_catalog.json"
        )
        self.catalog_path = candidate.resolve() if configured_catalog or candidate.exists() else None
        self.python = Path(python or sys.executable)
        self.drivers = drivers or {
            "autonomous_research.v1": AutonomousResearchDriver(),
            "generated_model_evolution.v1": EvolvingModelDriver(),
            "generated_model.v1": GeneratedModelDriver(),
            "reproduction_goal.v1": ReproductionDriver(),
            "paper_campaign.v1": PaperCampaignDriver(),
            **paper_step_drivers(self.root),
        }
        self.children = {}

    def profile(self):
        profile = Profile.from_dict(json.loads(self.profile_path.read_text()))
        if profile.mode != "science_expert":
            raise ValueError("Science goals require an expert-team profile")
        return profile

    def runtime_paths(self):
        spec = importlib.util.spec_from_file_location(
            "venus_goal_probe", self.root / "scripts/rsi/run_harness_probe.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.default_runtimes()

    def execution_catalog(self):
        return ExecutionCatalog(
            self.root, self.profile(), path=self.catalog_path, legacy_credential=self.credential_path, free_only=provider_online(), include_configured=True
        )

    def admission(self, profile=None, credential_path=None):
        profile = profile or self.profile()
        if profile.mode != "science_expert":
            raise ValueError("Science goals require an expert-team profile")
        from agent.harness.catalog import free_model

        from agent.harness.provider_config import online
        if online() and not free_model(profile.model):
            raise ValueError("Select a free model")
        resolve_credential(profile.model, credential_path)
        check = builtin_registry().preflight(profile, self.runtime_paths())
        if not check["ready"]:
            raise ValueError("Configured goal execution profile is unsupported")

    def prepared_profile(self, folder):
        return folder.parent / ".execution" / (folder.name + ".json")

    def frozen_profile(self, folder):
        return Profile.from_dict(GoalStore(folder).read()["definition"]["profile"])

    def frozen_credential_path(self, folder):
        meta = json.loads((folder.parent / (folder.name + ".json")).read_text())
        if "credential_toml" in meta:
            return Path(meta["credential_toml"]) if meta["credential_toml"] else None
        return self.credential_path  # Goals created before per-goal configuration.

    def catalog(self, include_cases=True):
        execution = None
        try:
            execution = self.execution_catalog().public(self.runtime_paths())
            ready = any(c["ready"] for c in execution["combinations"])
            profile = self.profile()
        except (OSError, ValueError, KeyError, TypeError):
            ready, profile = False, None
        return {
            "default_driver": "autonomous_research.v1",
            "available": ready,
            "model": profile.model.model if profile else None,
            "execution": execution,
            "drivers": [
                {
                    "id": d.id,
                    "title": d.title,
                    "criteria": list(d.criteria),
                    "cases": list(d.cases),
                    "maximum_budget": driver_limits(d),
                    "case_details": d.case_details(profile) if hasattr(d, "case_details") else {},
                }
                for d in self.drivers.values()
                if d.id == "autonomous_research.v1" or (include_cases and d.cases)
            ],
            "default_budget": {"max_cycles": 8, "model_calls": 192, "tool_calls": 96, "wall_seconds": 7200},
            "maximum_budget": {"max_cycles": 8, "model_calls": 192, "tool_calls": 96, "wall_seconds": 7200},
        }

    def generated_steps(self, session_id, case):
        return available_steps(self, session_id, case)

    def session_folder(self, session_id):
        if not isinstance(session_id, str) or not session_id or len(session_id) > 256:
            raise ValueError("Invalid session identity")
        return self.state_root / hashlib.sha256(session_id.encode()).hexdigest()

    def paths(self, session_id, goal_id):
        if not re.fullmatch(r"[0-9a-f]{32}", goal_id):
            raise ValueError("Invalid goal identity")
        base = self.session_folder(session_id)
        return base / goal_id, base / (goal_id + ".json")

    def metadata(self, session_id, goal_id):
        folder, path = self.paths(session_id, goal_id)
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError("Goal not found in this session")
        value = json.loads(path.read_text())
        if value["session_id"] != session_id:
            raise ValueError("Goal ownership mismatch")
        return folder, path, value

    @staticmethod
    def leased(folder, name):
        path = folder / name
        if not path.exists():
            return False
        with path.open("a") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
        return False

    def live(self, folder):
        if (folder / 'queue.json').is_file():
            from backend.web_v2.job_queue import JobQueue
            try:
                queued = JobQueue().get(json.loads((folder / 'queue.json').read_text())['id'])
                if queued['status'] in {'queued','running'}:return True
            except FileNotFoundError:pass
        if self.leased(folder, ".worker.lock") or self.leased(folder, ".runner.lock"):
            return True
        key = str(folder)
        child = self.children.get(key)
        if child is not None:
            if child.poll() is None:
                return True
            self.children.pop(key, None)
        p = folder / "worker.json"
        if p.is_file():
            identity = json.loads(p.read_text())
            if process_identity(identity["pid"]) == identity:
                return True
        return False

    def reconcile(self, folder):
        # Recover orphaned children before checking leases, including direct resume.
        if (folder / "queue.json").is_file():
            from backend.web_v2.job_queue import JobQueue
            JobQueue().get(json.loads((folder / "queue.json").read_text())["id"])
        store = GoalStore(folder)
        state = store.read()
        if (state["status"] not in {"active", "paused"} and state["inflight"] is None) or self.live(folder):
            return state
        # A surviving runner takes precedence even if its process record was lost.
        with (folder / ".runner.lock").open("a") as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return state

            def recover(s):
                if s["inflight"]:
                    record = {
                        **s["inflight"],
                        "status": "interrupted",
                        "summary": "Worker ended before checkpoint; reserved cost retained",
                    }
                    s["cycles"].append(record)
                    atomic_json(folder / f"cycle_{record['index']:03d}.json", record)
                    s["inflight"] = None
                if s["status"] == "active":
                    s["status"] = "interrupted"
                    s["events"].append({"event": "worker_missing", "at": time.time()})

            return store.mutate(recover)

    def public(self, session_id, goal_id):
        folder, _, meta = self.metadata(session_id, goal_id)
        if not (folder / "goal.json").is_file():
            return {
                "id": goal_id,
                "status": meta["status"],
                "objective": meta["spec"]["objective"],
                "driver": meta["spec"]["driver"],
                "worker_alive": False,
                "created_at": meta["created_at"],
            }
        schedule = None
        if (folder/'queue.json').is_file():
            from backend.web_v2.job_queue import JobQueue
            schedule = JobQueue().public(json.loads((folder/'queue.json').read_text())['id'])
        state = self.reconcile(folder)
        checks = (state.get("assessment") or {}).get("criteria", {})
        return {
            "id": goal_id,
            "status": state["status"],
            "queue": schedule,
            "driver": meta["spec"]["driver"],
            "objective": state["definition"]["goal"]["objective"],
            "budget": {k: v for k, v in state["definition"]["goal"].items() if k not in {"objective", "criteria"}},
            "usage": state["usage"],
            "model": state["definition"]["profile"]["model"]["model"],
            "execution": {
                "harness": state["definition"]["profile"]["harness"],
                "provider": state["definition"]["profile"]["model"]["provider"],
                "model": state["definition"]["profile"]["model"]["model"],
                "protocol": state["definition"]["profile"]["model"]["protocol"],
                "profile_sha256": digest(state["definition"]["profile"]),
            },
            "criteria": [
                {
                    "name": name,
                    "passed": checks.get(name, {}).get("passed", False),
                    "evidence_count": len(checks.get(name, {}).get("evidence", [])),
                }
                for name in state["definition"]["goal"]["criteria"]
            ],
            "cycles": [
                {
                    **{k: c.get(k) for k in ["index", "phase", "status", "summary", "usage", "generation"]},
                    "phase": c.get("phase")
                    or ("import" if meta["spec"]["driver"] == "paper_step_transfer.v1" and c["index"] == 0 else None),
                }
                for c in state["cycles"]
            ],
            "inflight_index": state["inflight"]["index"] if state["inflight"] else None,
            "worker_alive": self.live(folder),
            "created_at": meta["created_at"],
            "parameters": meta["spec"]["parameters"],
            "activity": goal_activity(folder, state),
            "project": project_summary(folder, state),
            "research": {**{k: state["progress"].get(k) for k in ["pending_interaction", "plan", "questions", "artifacts", "capabilities", "last_review", "inputs", "publication_reviews", "research_completed", "development", "deliveries", "selected_deliveries", "delivery_pending"]}, "pending_updates": len(state.get("pending_research_updates", [])), "live": research_live(folder, state)} if meta["spec"]["driver"] == "autonomous_research.v1" else None,
            "reproduction_recipes": public_reproduction_recipes(folder, state),
            "papers": public_paper_progress(folder, state),
            "code_step": public_step(folder, state),
            "capabilities": [
                {
                    "generation": g.get("generation", 1),
                    "model_id": g["model_id"],
                    "source_sha256": g.get("source_sha256"),
                    "experience_id": g.get("experience_id"),
                    "reused": bool(g.get("reuse_receipt")),
                }
                for g in state["progress"].get(
                    "generations",
                    [state["progress"]["accepted_model"]] if state["progress"].get("accepted_model") else [],
                )
            ],
        }

    def list(self, session_id):
        base = self.session_folder(session_id)
        if not base.exists():
            return []
        with file_lock(base / ".supervisor.lock"):
            return sorted(
                [self.public(session_id, p.stem) for p in base.glob("*.json")],
                key=lambda x: x["created_at"],
                reverse=True,
            )

    def environment(self, profile=None):
        env = {
            "PATH": str(self.python.parent) + ":/usr/bin:/bin",
            "PYTHONPATH": os.pathsep.join(dict.fromkeys([str(self.root / "src"), *[p for p in sys.path if p]])),
            "HOME": str(Path.home()),
            "LANG": "C.UTF-8",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        if profile is not None:
            name = profile.model.credential_env
            if os.environ.get(name):
                env[name] = os.environ[name]
        if os.environ.get("VENUS_EXECUTION_CATALOG"):
            env["VENUS_EXECUTION_CATALOG"] = os.environ["VENUS_EXECUTION_CATALOG"]
        from backend.web_v2.online_access import online
        if online():
            env.update(WEBUI_V2_MODE="online", VENUS_SCIENTIST_ONLINE="1", CUDA_VISIBLE_DEVICES="", VENUS_QUEUE_WAIT_TIMEOUT="3600")
            for name in ('TAVILY_API_KEY', 'SERPAPI_API_KEY', 'SERPAPI_KEY'):
                if os.environ.get(name): env[name] = os.environ[name]
        # A detached Scientist is a backend worker in the SAME deployment.
        # In particular its scheduler URL and signing-key directory must agree;
        # otherwise every queued tool fails before reaching the public service.
        from backend.web_v2.runtime_paths import AREAS
        runtime_keys = {*AREAS, 'VENUS_RUNTIME_ROOT', 'WEBUI_V2_MODE', 'MCP_HTTP_PORT',
                        'VENUS_EXECUTION_CATALOG', 'VENUS_MODEL_RUNTIMES_CONFIG',
                        'VENUS_CAPABILITY_DB', 'VENUS_SANDBOX_BACKEND',
                        'VENUS_SANDBOX_PYTHON', 'VENUS_SANDBOX_BWRAP', 'VENUS_SESSION_SANDBOX',
                        'TAVILY_API_KEY', 'SERPAPI_API_KEY', 'SERPAPI_KEY'}
        for name in runtime_keys:
            if os.environ.get(name): env[name] = os.environ[name]
        from agent.harness.provider_config import configuration_keys
        for name in configuration_keys():
            if not name.endswith("_API_KEY") and os.environ.get(name):
                env[name] = os.environ[name]
        return env

    def spawn(self, folder, driver, action):
        if self.live(folder):
            raise GoalConflict("Goal worker is still running")
        # The child inherits this lease before exec, closing the PID-record gap.
        # Closing the parent's fd must not explicitly unlock the shared lease.
        with (folder / ".worker.lock").open("a") as lease, (folder / "worker.log").open("ab") as log:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise GoalConflict("Goal worker is still running") from exc
            from backend.web_v2.job_queue import launch
            from backend.web_v2.online_access import online, SCIENCE_LIMITS
            limited = online()
            meta = json.loads((folder.parent / (folder.name + ".json")).read_text())
            queue_id, proc = launch(driver.run_command(self, folder, action),owner=meta.get("online_owner") or str(folder.parent.name),
                kind='scientist',resource='research',cwd=self.root,env=self.environment(self.frozen_profile(folder)),
                stdout=log,pass_fds=(lease.fileno(),),timeout=SCIENCE_LIMITS['wall_seconds'] if limited else int(os.getenv('VENUS_QUEUE_RESEARCH_TIMEOUT','2419200')),
                session_key='chat:'+folder.parent.name,
                metadata={'goal_id':folder.name, 'goal_folder':str(folder), 'session_id':folder.parent.name, 'mode':'scientist', 'harness':self.frozen_profile(folder).harness, **({'quota_scope':'chat:'+folder.parent.name} if limited else {})})
            atomic_json(folder/'queue.json',{'id':queue_id})
        self.children[str(folder)] = proc
        identity = process_identity(proc.pid) if not getattr(proc, 'identity', None) else None
        if identity:
            atomic_json(folder / "worker.json", identity)

    def create(self, session_id, spec, request_id, *, before_start=None, online_owner=None):
        from backend.web_v2.online_access import SCIENCE_LIMITS, online
        limited = online()
        if limited and (spec.get("driver", "autonomous_research.v1") != "autonomous_research.v1" or spec.get("parameters")):
            raise ValueError("Online Scientist supports autonomous research only")
        if limited and any(type(spec["budget"].get(k)) is not int or not 1 <= spec["budget"][k] <= v for k,v in SCIENCE_LIMITS.items()):
            raise ValueError("Online Scientist allows 2 iterations, 48 model calls, 24 tools and 15 minutes")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", request_id):
            raise ValueError("A stable request identity is required")
        spec = {"driver": "autonomous_research.v1", "parameters": {}, **spec}
        required = {"driver", "parameters", "objective", "budget"}
        if not required <= set(spec) or set(spec) - required - {"execution", "attachments"} or spec["driver"] not in self.drivers:
            raise ValueError("Unknown scientific goal driver")
        if not isinstance(spec["objective"], str) or not 5 <= len(spec["objective"].strip()) <= 8000:
            raise ValueError("Provide a concrete research objective")
        maxima = driver_limits(self.drivers[spec["driver"]])
        if set(spec["budget"]) != set(maxima) or any(
            type(spec["budget"][k]) is not int or not 1 <= spec["budget"][k] <= limit for k, limit in maxima.items()
        ):
            raise ValueError("Goal budget exceeds the configured limits")
        from backend.web_v2.research_files import resolve_inputs, stage_inputs
        attachments = spec.get("attachments", [])
        if attachments and spec["driver"] != "autonomous_research.v1":
            raise ValueError("Attachments require autonomous research")
        resolve_inputs(self.session_folder(session_id), attachments)
        driver = self.drivers[spec["driver"]]
        gid = hashlib.sha256(request_id.encode()).hexdigest()[:32]
        folder, meta_path = self.paths(session_id, gid)
        base = folder.parent
        base.mkdir(parents=True, exist_ok=True)
        with file_lock(base / ".supervisor.lock"):
            if meta_path.exists():
                _, _, old = self.metadata(session_id, gid)
                if digest(old["spec"]) != digest(spec):
                    raise GoalConflict("Request identity already belongs to a different goal")
                return self.public(session_id, gid)
            if limited and any(base.glob("*.json")):
                raise GoalConflict("Start a new Scientist conversation for another study")
            command = driver.prepare_command(self, folder, spec)
            if any(
                g["status"] in NONTERMINAL or g["worker_alive"]
                for g in (self.public(session_id, p.stem) for p in base.glob("*.json"))
            ):
                raise GoalConflict("This session already has an unfinished goal")
            profile, model = self.execution_catalog().select(spec.get("execution"))
            self.admission(profile, model.credential_toml)
            if before_start: before_start()
            snapshot = self.prepared_profile(folder)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(snapshot, profile.public())
            meta = {
                "session_id": session_id,
                "online_owner": online_owner,
                "spec": spec,
                "created_at": time.time(),
                "status": "preparing",
                "credential_toml": str(model.credential_toml) if model.credential_toml else None,
                "profile_sha256": digest(profile.public()),
            }
            atomic_json(meta_path, meta)
            try:
                with (base / (gid + ".prepare.log")).open("wb") as log:
                    subprocess.run(
                        command,
                        cwd=self.root,
                        env=self.environment(profile),
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=60,
                    )
                if digest(self.frozen_profile(folder).public()) != meta["profile_sha256"]:
                    raise ValueError("Prepared execution profile differs from selected profile")
                if attachments:
                    inputs = stage_inputs(base, folder / "workspace", attachments)
                    GoalStore(folder).mutate(lambda s: s["progress"].update(inputs=inputs))
                self.spawn(folder, driver, "run")
            except Exception as exc:
                if limited:
                    from backend.web_v2.online_access import AccessStore
                    AccessStore().release_science("chat:" + session_id)
                meta.update(status="prepare_failed", error_type=type(exc).__name__)
                atomic_json(meta_path, meta)
                raise RuntimeError("Goal could not start; preparation diagnostics retained") from exc
            meta["status"] = "started"
            atomic_json(meta_path, meta)
            return self.public(session_id, gid)

    def control(self, session_id, goal_id, action):
        folder, _, _ = self.metadata(session_id, goal_id)
        with file_lock(folder.parent / ".supervisor.lock"):
            folder, _, meta = self.metadata(session_id, goal_id)
            state = self.reconcile(folder)
            if action in {'pause','cancel'} and (folder/'queue.json').is_file():
                from backend.web_v2.job_queue import JobQueue
                q=JobQueue();qid=json.loads((folder/'queue.json').read_text())['id']
                if action=='cancel' or q.get(qid)['status']=='queued':q.cancel(qid)
            if action == "pause":
                GoalStore(folder).pause()
            elif action == "resume":
                if self.live(folder):
                    raise GoalConflict("Wait for the current worker to stop before resuming")
                if state["status"] not in {"paused", "interrupted", "waiting_for_input"}:
                    raise GoalConflict("Only a stopped unfinished goal can resume")
                self.admission(self.frozen_profile(folder), self.frozen_credential_path(folder))
                self.spawn(folder, self.drivers[meta["spec"]["driver"]], "resume")
            elif action == "publish":
                raise GoalConflict("Request platform publication by email; only the platform operator can publish reviewed models")
            elif action == "cancel":
                if state["status"] not in {"completed", "budget_exhausted", "cancelled"}:

                    def cancel(s):
                        s["status"] = "cancelled"
                        s["events"].append({"event": "user_cancelled", "at": time.time()})

                    GoalStore(folder).mutate(cancel)
            else:
                raise ValueError("Unknown goal control")
            return self.public(session_id, goal_id)

    def provide_input(self, session_id, goal_id, text, plan=None, attachments=None, interaction=None):
        from agent.harness.research_updates import queue_revision, validate_revision
        from backend.web_v2.research_files import stage_inputs
        validate_revision(text or ("User response" if interaction else ""), plan, attachments)
        folder, _, meta = self.metadata(session_id, goal_id)
        with file_lock(folder.parent / ".supervisor.lock"):
            state = self.reconcile(folder)
            if meta["spec"]["driver"] != "autonomous_research.v1" or state["status"] not in NONTERMINAL:
                raise GoalConflict("This research has ended")
            inputs = stage_inputs(folder.parent, folder / "workspace", attachments or [])
            queue_revision(GoalStore(folder), text, plan, inputs, interaction)
            if not self.live(folder) and state["status"] in {"waiting_for_input", "paused", "interrupted"}:
                self.admission(self.frozen_profile(folder), self.frozen_credential_path(folder))
                self.spawn(folder, self.drivers[meta["spec"]["driver"]], "resume")
            return self.public(session_id, goal_id)

    def has_running_goal(self, session_id):
        return any(g["status"] == "active" or g["worker_alive"] for g in self.list(session_id))

    def has_unfinished_goal(self, session_id):
        return any(g["status"] in NONTERMINAL or g["worker_alive"] for g in self.list(session_id))


service = ScienceGoalService()
