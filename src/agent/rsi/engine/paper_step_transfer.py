"""Fresh-goal import, independent acceptance and reuse of generated code."""

import asyncio
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from agent.harness.contracts import file_hash
from agent.science_goal import GoalStore, validate_assessment

from .generated_step_library import GeneratedStepLibrary, validate_transfer
from .paper_step import PaperStepLifecycle, digest
from agent.rsi.campaigns.paper_step_goal import ROOT, PaperStepGoal
from .paper_step_sessions import read
from .source_snapshot import research_implementation_paths
from .storage import atomic_json, file_lock, safe_child

CRITERIA = ("verified_source_imported", "recipient_acceptance_passed", "fresh_inputs_reused")


def create_transfer_goal(
    root, definition, profile, spec, library, entry_id=None, *, condition="science_goal", tool_context=None
):
    if tool_context is not None:
        from .scientist_tool_context import validate_context

        validate_context(tool_context, definition)
    root = library.goal(root)
    if tuple(spec.criteria) != CRITERIA:
        raise ValueError("Transfer requires import, independent acceptance and fresh reuse")
    if entry_id is None:
        matches = library.find(definition)
        if len(matches) != 1:
            raise ValueError("Transfer needs exactly one compatible match or an explicit entry ID")
        entry_id = matches[0]["entry_id"]
    origin = library.resolve(entry_id)
    validate_transfer(origin, definition)
    if root == Path(origin["origin_goal"]):
        raise ValueError("Transfer needs a distinct goal")
    admission = {"kind": "generated_paper_step_transfer.v1", "definition_sha256": digest(definition)}
    sources = research_implementation_paths(ROOT)
    imported = {
        "library": str(library.root),
        "entry_id": entry_id,
        "origin_goal": origin["origin_goal"],
        "origin_sha256": digest(origin),
        "novelty_policy": "complete_inputs_disjoint_v1",
        "claim_limit": "Exact-input novelty only; not entity-level or predictive generalization",
    }
    protocol = {
        "admission": admission,
        "import": imported,
        "implementation": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        "cycle_wall_seconds": profile.budget.wall_seconds
        + len(definition["cases"]["acceptance"]) * definition["budget"]["operation_seconds"]
        + 30,
    }
    if tool_context is not None:
        protocol["scientist_tool_context"] = tool_context
    store = GoalStore.create(
        root, spec, profile, condition=condition, admission={**admission, "protocol_sha256": digest(protocol)}
    )
    atomic_json(root / "step_definition.json", definition)
    atomic_json(root / "protocol.json", protocol)
    atomic_json(root / "import.json", imported)
    PaperStepLifecycle(root / "host_private", definition)
    for p in sources:
        target = root / "implementation" / p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
    return store


def accept_import(store, folder):
    """Host-only; every transferred source gets new recipient acceptance."""
    driver = TransferredStepGoal(store, None, {})
    folder = safe_child(store.root, str(Path(folder).absolute().relative_to(store.root)))
    origin = driver.origin()
    entry = origin["entry_id"]
    with file_lock(folder / ".import_acceptance.lock"):
        reservation = folder / "host_acceptance_reservation.json"
        if reservation.exists():
            raise ValueError("Import acceptance was already reserved; reconcile retained evidence")
        atomic_json(reservation, {"entry_id": entry, "origin_sha256": digest(origin)})
        identity = driver.lifecycle.propose(origin["packet"]["source_code"], declared_origin="library_import:" + entry)
        atomic_json(folder / "host_candidate.json", {"candidate_id": identity, "entry_id": entry})
        assessment = driver.lifecycle.assess(identity)
        result = {
            "candidate_id": identity,
            "assessment": assessment,
            "library_entry_id": entry,
            "source_provenance": origin["development_proof"],
            "imported_source": True,
            "new_code_generated": False,
        }
        if assessment["passed"]:
            result["capability_id"] = driver.lifecycle.register(identity)
        atomic_json(folder / "host_acceptance.json", result)
        return result


async def import_worker(store, folder):
    command = [
        sys.executable,
        str(ROOT / "scripts/rsi/paper_step_import_worker.py"),
        "--goal",
        str(store.root),
        "--folder",
        str(folder),
    ]
    with (folder / "host_worker.log").open("xb") as log:
        proc = await asyncio.create_subprocess_exec(
            *command,
            stdout=log,
            stderr=log,
            start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
        )
        atomic_json(folder / "host_worker.json", {"pid": proc.pid, "command": command, "started_unix": time.time()})
        try:
            code = await proc.wait()
        except BaseException:
            if proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
            atomic_json(folder / "host_worker_termination.json", {"interrupted": True, "returncode": proc.returncode})
            raise
    atomic_json(folder / "host_worker_exit.json", {"returncode": code})
    if code:
        raise RuntimeError("Import acceptance failed; inspect retained worker log and reservations")
    return read(folder / "host_acceptance.json")


class TransferredStepGoal(PaperStepGoal):
    def __init__(self, store, credential, runtimes):
        self.store, self.credential, self.runtimes = store, credential, runtimes
        self.protocol = read(store.root / "protocol.json")
        self.definition = read(store.root / "step_definition.json")
        admission = {"kind": "generated_paper_step_transfer.v1", "definition_sha256": digest(self.definition)}
        if self.protocol["admission"] != admission or store.read()["definition"].get("admission") != {
            **admission,
            "protocol_sha256": digest(self.protocol),
        }:
            raise ValueError("Transfer admission changed")
        self.check_sources()
        self.origin()
        self.lifecycle = PaperStepLifecycle(store.root / "host_private", self.definition)
        if "scientist_tool_context" in self.protocol:
            from .research_tool_lifecycle import ResearchToolLifecycle

            self.lifecycle = ResearchToolLifecycle(
                store.root / "host_private",
                self.definition,
                self.protocol["scientist_tool_context"]["contract"],
            )

    def origin(self):
        imported = read(self.store.root / "import.json")
        if imported != self.protocol["import"]:
            raise ValueError("Bound library import changed")
        library = GeneratedStepLibrary(imported["library"])
        library.goal(self.store.root)
        origin = library.resolve(imported["entry_id"])
        if origin["origin_goal"] != imported["origin_goal"] or digest(origin) != imported["origin_sha256"]:
            raise ValueError("Bound library source changed")
        validate_transfer(origin, self.definition)
        return origin

    def evaluate(self, state):
        self.check_sources()
        origin = self.origin()
        progress = state["progress"]
        checks = {k: {"passed": False, "evidence": []} for k in CRITERIA}
        checks[CRITERIA[0]] = {"passed": True, "evidence": [self.evidence(self.store.root / "import.json")]}
        if (
            progress.get("hold")
            and self.evidence(safe_child(self.store.root, progress["hold"]["path"])) != progress["hold"]
        ):
            raise ValueError("Held import evidence changed")
        identity = progress.get("capability_id")
        if identity is None:
            return {"criteria": checks}
        packet = self.lifecycle.capability(identity)
        folder = safe_child(self.store.root, progress["development_folder"])
        receipt = read(folder / "host_acceptance.json")
        if (
            packet["source_code"] != origin["packet"]["source_code"]
            or packet["declared_origin"] != "library_import:" + origin["entry_id"]
        ):
            raise ValueError("Recipient registered source differs from immutable import")
        expected = {
            "candidate_id": packet["verification"]["candidate_id"],
            "assessment": packet["verification"],
            "library_entry_id": origin["entry_id"],
            "source_provenance": origin["development_proof"],
            "imported_source": True,
            "new_code_generated": False,
            "capability_id": identity,
        }
        if receipt != expected:
            raise ValueError("Recipient acceptance does not match imported source")
        if read(folder / "host_acceptance_reservation.json") != {
            "entry_id": origin["entry_id"],
            "origin_sha256": digest(origin),
        }:
            raise ValueError("Import acceptance reservation changed")
        checks[CRITERIA[1]] = {"passed": True, "evidence": [self.evidence(folder / "host_acceptance.json")]}
        reuse = progress.get("reuse", {})
        required = {c["id"] for c in self.definition["cases"]["reuse"]}
        if set(reuse) - required:
            raise ValueError("Unadmitted recipient reuse")
        sessions = set(origin["sessions"])
        evidence = []
        for case, relative in reuse.items():
            folder = safe_child(self.store.root, relative)
            proof = self.phase_proof(state, folder)
            actual = self.lifecycle.verify_reuse(identity, case)
            if (
                not actual["passed"]
                or read(folder / "host_acceptance.json") != {"verification": actual, "source_provenance": proof}
                or proof["session_id"] in sessions
            ):
                raise ValueError("Recipient reuse is not independently verified in a fresh session")
            sessions.add(proof["session_id"])
            evidence.append(self.evidence(folder / "host_acceptance.json"))
        if set(reuse) == required:
            checks[CRITERIA[2]] = {"passed": True, "evidence": evidence}
        return {"criteria": checks}

    async def cycle(self, index, profile, state, context):
        if state["progress"].get("capability_id") or state["progress"].get("hold"):
            return await super().cycle(index, profile, state, context)
        started = time.monotonic()
        self.check_sources()
        self.origin()
        progress = dict(state["progress"])
        for path in (self.store.root / "episodes").glob("cycle_*/host_acceptance_reservation.json"):
            progress["hold"] = self.evidence(path)
            return {
                "status": "held",
                "needs_input": True,
                "summary": "An existing import acceptance must be reconciled before retrying.",
                "progress": progress,
                "usage": {"model_calls": 0, "tool_calls": 0, "wall_seconds": time.monotonic() - started},
            }
        folder = self.store.root / "episodes" / f"cycle_{index:03d}"
        folder.mkdir(parents=True, exist_ok=False)
        result = {
            "status": "import_rejected",
            "progress": progress,
            "summary": "Recipient acceptance failed; immutable source and evidence retained",
            "needs_input": True,
        }
        try:
            accepted = await import_worker(self.store, folder)
            if accepted["assessment"]["passed"]:
                progress.update(
                    capability_id=accepted["capability_id"], development_folder=str(folder.relative_to(self.store.root))
                )
                result.update(
                    status="completed",
                    needs_input=False,
                    summary="Imported source independently accepted on recipient references; fresh sessions still required",
                )
            else:
                progress["hold"] = self.evidence(folder / "host_acceptance.json")
        except Exception as exc:
            marker = folder / "host_worker_exit.json"
            if not marker.exists():
                marker = self.store.root / "import.json"
            progress["hold"] = self.evidence(marker)
            result.update(
                error_type=type(exc).__name__,
                summary="Import verification interrupted; retained evidence requires reconciliation",
            )
        result["usage"] = {"model_calls": 0, "tool_calls": 0, "wall_seconds": time.monotonic() - started}
        atomic_json(folder / "outcome.json", result)
        return result


def validate_completed_transfer(folder, library):
    store = GoalStore(library.goal(folder))
    state = store.read()
    driver = TransferredStepGoal(store, None, {})
    if (
        state["status"] != "completed"
        or state["inflight"] is not None
        or driver.protocol["import"]["library"] != str(library.root)
    ):
        raise ValueError("Only completed transfers in this library may republish their original capability")
    assessment = driver.evaluate(state)
    if assessment != state["assessment"] or not validate_assessment(store, state, assessment):
        raise ValueError("Transfer completion evidence changed")
    return driver.protocol["import"]["entry_id"]
