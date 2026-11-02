"""Persistent generated-paper-step development, host acceptance and fresh reuse."""

import asyncio
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from agent.harness import Profile
from agent.harness.contracts import file_hash
from agent.harness.events import audit_requests, audit_trace
from agent.science_goal import GoalStore

from agent.rsi.engine.paper_step import PaperStepLifecycle, digest, validate_definition
from agent.rsi.engine.paper_step_sessions import PHASE_TOOLS, execute_phase, prepare_phase, read, verify_phase
from agent.rsi.engine.provider_access import access_wait
from agent.rsi.engine.source_snapshot import research_implementation_paths
from agent.rsi.engine.storage import atomic_json, safe_child

ROOT = Path(__file__).resolve().parents[4]
CRITERIA = ("submitted_source_verified", "accepted_step_registered", "fresh_inputs_reused")


def create_step_goal(root, definition, profile, spec, *, condition="science_goal", tool_context=None):
    validate_definition(definition)
    if tool_context is not None:
        from agent.rsi.engine.scientist_tool_context import validate_context

        validate_context(tool_context, definition)
    if tuple(spec.criteria) != CRITERIA:
        raise ValueError("Paper step goal must require source, registration and fresh reuse")
    admission = {"kind": "generated_paper_step_goal.v1", "definition_sha256": digest(definition)}
    sources = list(dict.fromkeys(research_implementation_paths(ROOT)))
    implementation = {str(p.relative_to(ROOT)): file_hash(p) for p in sources}
    protocol = {
        "admission": admission,
        "implementation": implementation,
        "cycle_wall_seconds": profile.budget.wall_seconds
        + len(definition["cases"]["acceptance"]) * definition["budget"]["operation_seconds"]
        + 30,
        "scope": "Generated numerical paper steps; independent acceptance and fresh-input reuse. No complete-paper, biological-discovery or RSI-efficacy claim.",
    }
    if tool_context is not None:
        protocol["scientist_tool_context"] = tool_context
    store = GoalStore.create(
        root, spec, profile, condition=condition, admission={**admission, "protocol_sha256": digest(protocol)}
    )
    atomic_json(store.root / "step_definition.json", definition)
    PaperStepLifecycle(store.root / "host_private", definition)
    for source in sources:
        relative = str(source.relative_to(ROOT))
        target = store.root / "implementation" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    atomic_json(store.root / "protocol.json", protocol)
    return store


async def host_acceptance_worker(store, folder, phase, *, development_session=None):
    command = [
        sys.executable,
        str(ROOT / "scripts/rsi/paper_step_host_worker.py"),
        "--goal",
        str(store.root),
        "--folder",
        str(folder),
        "--phase",
        phase,
    ]
    if development_session is not None:
        command += ["--development-session", development_session]
    with (folder / "host_worker.log").open("xb") as log:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=log,
            stderr=log,
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )
        atomic_json(folder / "host_worker.json", {"pid": proc.pid, "started_unix": time.time(), "command": command})
        try:
            code = await proc.wait()
        except BaseException:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
            atomic_json(folder / "host_worker_termination.json", {"returncode": proc.returncode, "interrupted": True})
            raise
    atomic_json(folder / "host_worker_exit.json", {"returncode": code})
    if code != 0:
        raise RuntimeError("Host acceptance worker failed; retained worker log and reservations require inspection")
    return read(folder / "host_acceptance.json")


class PaperStepGoal:
    def __init__(self, store, credential, runtimes):
        self.store, self.credential, self.runtimes = store, credential, runtimes
        self.protocol = read(store.root / "protocol.json")
        self.definition = read(store.root / "step_definition.json")
        admission = {"kind": "generated_paper_step_goal.v1", "definition_sha256": digest(self.definition)}
        if self.protocol["admission"] != admission or store.read()["definition"].get("admission") != {
            **admission,
            "protocol_sha256": digest(self.protocol),
        }:
            raise ValueError("Paper step goal admission changed")
        self.check_sources()
        self.lifecycle = PaperStepLifecycle(store.root / "host_private", self.definition)
        if "scientist_tool_context" in self.protocol:
            from agent.rsi.engine.research_tool_lifecycle import ResearchToolLifecycle

            self.lifecycle = ResearchToolLifecycle(
                store.root / "host_private",
                self.definition,
                self.protocol["scientist_tool_context"]["contract"],
            )

    def check_sources(self):
        for relative, sha in self.protocol["implementation"].items():
            if file_hash(safe_child(ROOT, relative)) != sha:
                raise ValueError("Goal implementation changed; an explicit new allocation is required")

    def evidence(self, path):
        return {"path": str(path.relative_to(self.store.root)), "sha256": file_hash(path)}

    def phase_proof(self, state, folder):
        proof = verify_phase(folder)
        if "scientist_tool_context" in self.protocol:
            from agent.rsi.engine.scientist_tool_context import verify_context

            verify_context(folder, self.protocol["scientist_tool_context"], self.definition)
        base = Profile.from_dict(state["definition"]["profile"])
        actual = Profile.from_dict(read(folder / "runtime/profile.lock.json")["profile"])
        original = base.public()
        observed = actual.public()
        original.pop("budget")
        observed.pop("budget")
        if original != observed:
            raise ValueError("Goal mode, harness or model binding changed")
        for name in ["temperature", "thinking", "reasoning_effort", "output_tokens", "request_seconds"]:
            if getattr(actual.budget, name) != getattr(base.budget, name):
                raise ValueError("Goal inference settings changed")
        for name in ["model_calls", "tool_calls", "wall_seconds"]:
            if getattr(actual.budget, name) > getattr(base.budget, name):
                raise ValueError("Goal phase exceeds admitted inference limits")
        return proof

    def evaluate(self, state):
        self.check_sources()
        checks = {key: {"passed": False, "evidence": []} for key in CRITERIA}
        progress = state["progress"]
        if progress.get("hold"):
            held = progress["hold"]
            if self.evidence(safe_child(self.store.root, held["path"])) != held:
                raise ValueError("Held paper-step evidence changed")
        identity = progress.get("capability_id")
        if identity is None:
            return {"criteria": checks}
        packet = self.lifecycle.capability(identity)
        folder = safe_child(self.store.root, progress["development_folder"])
        proof = self.phase_proof(state, folder)
        receipt = read(folder / "host_acceptance.json")
        if (
            receipt["source_provenance"] != proof
            or receipt["capability_id"] != identity
            or receipt["candidate_id"] != packet["verification"]["candidate_id"]
            or receipt["assessment"] != packet["verification"]
        ):
            raise ValueError("Registered source does not match the independently accepted session")
        for key in CRITERIA[:2]:
            checks[key] = {"passed": True, "evidence": [self.evidence(folder / "host_acceptance.json")]}
        evidence = []
        sessions = {proof["session_id"]}
        reuse = progress.get("reuse", {})
        expected = {c["id"] for c in self.definition["cases"]["reuse"]}
        if set(reuse) - expected:
            raise ValueError("Unadmitted reuse case in goal progress")
        for case_id, relative in reuse.items():
            folder = safe_child(self.store.root, relative)
            proof = self.phase_proof(state, folder)
            receipt = read(folder / "host_acceptance.json")
            actual = self.lifecycle.verify_reuse(identity, case_id)
            if (
                not actual["passed"]
                or receipt != {"verification": actual, "source_provenance": proof}
                or proof["session_id"] in sessions
            ):
                raise ValueError("Fresh reuse evidence or session identity differs")
            sessions.add(proof["session_id"])
            evidence.append(self.evidence(folder / "host_acceptance.json"))
        if set(reuse) == expected:
            checks[CRITERIA[2]] = {"passed": True, "evidence": evidence}
        return {"criteria": checks}

    async def cycle(self, index, profile, state, context):
        started = time.monotonic()
        progress = dict(state["progress"])
        folder = self.store.root / "episodes" / f"cycle_{index:03d}"
        if progress.get("hold"):
            self.evaluate(state)
            return {
                "status": "held",
                "needs_input": True,
                "summary": "Retained computation or incomplete evidence requires reconciliation; no code or model call was repeated.",
                "progress": progress,
                "usage": {"model_calls": 0, "tool_calls": 0, "wall_seconds": time.monotonic() - started},
            }
        # A lost goal checkpoint cannot authorize another expensive computation.
        known = {progress.get("development_folder"), *progress.get("reuse", {}).values()}
        for previous in (self.store.root / "episodes").glob("cycle_*"):
            rel = str(previous.relative_to(self.store.root))
            if (
                rel not in known
                and not (previous / "outcome.json").exists()
                and any(
                    (previous / name).exists()
                    for name in [
                        "workspace/step_candidate.json",
                        "workspace/step_reuse_reservation.json",
                        "host_acceptance_reservation.json",
                    ]
                )
            ):
                marker = next(
                    previous / name
                    for name in [
                        "workspace/step_candidate.json",
                        "workspace/step_reuse_reservation.json",
                        "host_acceptance_reservation.json",
                    ]
                    if (previous / name).exists()
                )
                progress["hold"] = self.evidence(marker)
                return {
                    "status": "held",
                    "needs_input": True,
                    "summary": "Uncheckpointed paper-step artifacts retained for reconciliation.",
                    "progress": progress,
                    "usage": {"model_calls": 0, "tool_calls": 0, "wall_seconds": time.monotonic() - started},
                }
        identity = progress.get("capability_id")
        phase = "reuse" if identity else "develop"
        case_id = None
        if (
            not identity
            and len(list((self.lifecycle.root / "attempts").glob("*/reservation.json")))
            >= self.definition["budget"]["candidates"]
        ):
            progress["hold"] = self.evidence(self.store.root / "step_definition.json")
            return {
                "status": "candidate_budget_exhausted",
                "needs_input": True,
                "summary": "The admitted candidate budget is exhausted; no further model call was made.",
                "progress": progress,
                "usage": {"model_calls": 0, "tool_calls": 0, "wall_seconds": time.monotonic() - started},
            }
        if identity:
            case_id = next(
                c["id"] for c in self.definition["cases"]["reuse"] if c["id"] not in progress.get("reuse", {})
            )
        prepare_phase(self.lifecycle, folder, phase, capability_id=identity, case_id=case_id)
        if "scientist_tool_context" in self.protocol:
            from agent.rsi.engine.scientist_tool_context import attach_context

            attach_context(folder, self.protocol["scientist_tool_context"], self.definition)
        declaration = read(folder / "phase.json")
        declaration["prompt"] += "\n\n" + context
        atomic_json(folder / "phase.json", declaration)
        run = await execute_phase(folder, profile, self.runtimes, credential=self.credential)
        trace = audit_trace(folder / "runtime/events.jsonl", PHASE_TOOLS[phase], profile.budget.tool_calls)
        requests = audit_requests(folder / "runtime/model_requests.jsonl")
        waiting = access_wait(folder / "runtime", self.store.root)
        outcome = {
            "phase": phase,
            "status": "failed",
            "progress": progress,
            "summary": "Paper step session did not pass; all source attempts and usage retained",
        }
        if trace["passed"] and (requests["passed"] or waiting):
            outcome["usage"] = {
                "model_calls": requests["model_calls"],
                "tool_calls": trace["tool_calls"],
                "wall_seconds": 0.0,
            }
        if run.get("status") == "completed" and run.get("audit", {}).get("passed"):
            self.phase_proof(state, folder)
            try:
                if phase == "develop":
                    result = await host_acceptance_worker(self.store, folder, phase)
                    if result["assessment"]["passed"]:
                        progress.update(
                            capability_id=result["capability_id"],
                            development_folder=str(folder.relative_to(self.store.root)),
                        )
                        outcome.update(
                            status="completed",
                            summary="Generated paper-step source verified and independently registered; fresh-input reuse remains",
                        )
                    else:
                        outcome.update(
                            status="candidate_rejected",
                            summary="Independent numerical acceptance rejected this source; validation outputs remain private",
                        )
                else:
                    development = read(
                        safe_child(self.store.root, progress["development_folder"]) / "host_acceptance.json"
                    )
                    result = await host_acceptance_worker(
                        self.store, folder, phase, development_session=development["source_provenance"]["session_id"]
                    )
                    if result["verification"]["passed"]:
                        progress["reuse"] = {
                            **progress.get("reuse", {}),
                            case_id: str(folder.relative_to(self.store.root)),
                        }
                        outcome.update(
                            status="completed",
                            summary="Registered paper step executed and independently verified on a fresh admitted input",
                        )
                    else:
                        progress["hold"] = self.evidence(folder / "host_acceptance.json")
                        outcome.update(
                            needs_input=True,
                            summary="Fresh execution failed numerical acceptance; source and failed output retained without recomputation",
                        )
            except Exception as exc:
                marker = folder / "host_worker_exit.json"
                if not marker.exists():
                    marker = folder / "phase.json"
                progress["hold"] = self.evidence(marker)
                outcome.update(
                    needs_input=True,
                    error_type=type(exc).__name__,
                    summary="Host acceptance did not finish; retained artifacts require reconciliation",
                )
        else:
            outcome.update(waiting)
            marker = next(
                (
                    folder / name
                    for name in ["workspace/step_candidate.json", "workspace/step_reuse_reservation.json"]
                    if (folder / name).exists()
                ),
                None,
            )
            if marker is not None or not trace["passed"]:
                progress["hold"] = self.evidence(marker or folder / "runtime/run.json")
                outcome.update(
                    needs_input=True,
                    summary="Incomplete session evidence or completed computation retained; no automatic resubmission",
                )
        if "usage" in outcome:
            outcome["usage"]["wall_seconds"] = time.monotonic() - started
        atomic_json(folder / "outcome.json", outcome)
        return outcome
