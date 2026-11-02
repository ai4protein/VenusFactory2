# Product integration: actual detached computations, explicitly simulated models.
import importlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import pytest
from test_web_science_goals import api_client as api_client

from agent.harness import Budget, ModelBinding, Profile
from agent.rsi.engine.paper_step_catalog import catalog
from agent.rsi.engine.storage import atomic_json, sha256_file
from agent.science_goal import GoalStore
from backend.web_v2.paper_step_goals import PaperStepDriver, public_step
from backend.web_v2.science_goals import ROOT, GoalConflict, ScienceGoalService


def wait_for(fn):
    # This checks detached execution and publication, not a startup latency SLA.
    # Allow cold interpreter startup plus the separately bounded audit phases.
    until = time.monotonic() + 120
    while time.monotonic() < until:
        value = fn()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("Detached numerical fixture did not reach expected state")


def install_task(root, name, offset=0):
    reference = root / "private_reference.txt"
    if not reference.exists():
        reference.write_text("PRIVATE_REFERENCE_SENTINEL")
    definition = {
        "schema": 1,
        "paper": {"title": "Numerical integration fixture", "doi": "test:fixture"},
        "interface": {"id": "sum.v1", "description": "List of numbers to a sum object"},
        "scope": {"covered_steps": ["Summation"], "excluded_steps": ["Biological discovery"]},
        "references": {str(reference): sha256_file(reference)},
        "budget": {"candidates": 2, "operation_seconds": 30},
        "tolerance": {"atol": 1e-12, "rtol": 0},
        "cases": {},
    }
    for phase, values in [("development", [1, 2]), ("acceptance", [3, 5]), ("reuse", [7, 11])]:
        inputs = [v + offset for v in values]
        definition["cases"][phase] = [{"id": phase, "inputs": inputs, "expected": {"sum": sum(inputs)}}]
    path = root / "configs/rsi/paper_step_definitions" / f"{name}.json"
    atomic_json(path, definition)
    manifest = root / "configs/rsi/paper_steps.json"
    value = json.loads(manifest.read_text()) if manifest.exists() else {"schema": 1, "tasks": {}}
    value["tasks"][name] = {
        "definition": str(path.relative_to(root)),
        "sha256": sha256_file(path),
        "label": {"en": name, "zh": name},
    }
    atomic_json(manifest, value)


@pytest.fixture
def service(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    install_task(root, "source_sum")
    install_task(root, "recipient_sum", 100)
    profile = Profile(
        "science_expert",
        "langgraph",
        ModelBinding("test_fixture", "simulated", "https://example.org/v1", credential_env="PAPER_STEP_FIXTURE_KEY"),
        Budget(wall_seconds=60, model_calls=12, tool_calls=8),
    )
    profile_path = root / "profile.json"
    atomic_json(profile_path, profile.public())
    monkeypatch.setenv("PAPER_STEP_FIXTURE_KEY", "unused-synthetic-provider-key")
    monkeypatch.syspath_prepend(str(ROOT / "scripts/rsi"))
    probe = importlib.import_module("run_harness_probe")
    wrapper = root / "scripts/rsi/run_web_paper_step.py"
    wrapper.parent.mkdir(parents=True)
    wrapper.write_text(f"""import asyncio,sys
from pathlib import Path
sys.path.insert(0,{str(ROOT / "src")!r})
sys.path.insert(0,{str(ROOT / "scripts/rsi")!r})
sys.path.insert(0,{str(ROOT / "tests/rsi")!r})
import run_web_paper_step as entry
from agent.rsi import paper_step_goal
from agent.rsi.engine.paper_step_sessions import read
from test_paper_step_tools import simulated_session
entry.ROOT=Path({str(root)!r})
async def execute(folder,profile,runtimes,credential):
    declaration=read(folder/'phase.json')
    task=simulated_session(folder,phase=declaration['phase'],capability_id=declaration['capability_id'],session_id='simulated-'+folder.parent.parent.name+'-'+folder.name,profile=profile)
    return read(task.output/'run.json')
paper_step_goal.execute_phase=execute
asyncio.run(entry.main(entry.parser().parse_args()))
""")
    svc = ScienceGoalService(
        root,
        state_root=tmp_path / "goals",
        profile_path=profile_path,
        python=sys.executable,
        drivers={d.id: d for d in [PaperStepDriver(root), PaperStepDriver(root, True)]},
    )
    svc.runtime_paths = probe.default_runtimes
    yield svc
    for folder, proc in list(svc.children.items()):
        if proc.poll() is None:
            store = GoalStore(Path(folder))
            if (store.root / "goal.json").exists():
                store.mutate(lambda s: s.update(status="cancelled"))
            try:
                proc.wait(timeout=15)
            except Exception:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=5)


def request(transfer=False, entry=None):
    return {
        "driver": "paper_step_transfer.v1" if transfer else "paper_step.v1",
        "parameters": {"case": "recipient_sum", "entry_id": entry} if transfer else {"case": "source_sum"},
        "objective": "Implement or reuse the admitted scientific computation and verify fresh inputs",
        "budget": {"max_cycles": 4, "model_calls": 48, "tool_calls": 32, "wall_seconds": 600},
    }


def complete(service, sid, gid):
    def check():
        value = service.public(sid, gid)
        if value["status"] in {"prepare_failed", "failed", "cancelled", "interrupted", "waiting_for_input", "budget_exhausted"}:
            folder, _ = service.paths(sid, gid)
            raise AssertionError((value, (folder / "worker.log").read_text()))
        return value if value["status"] == "completed" and not value["worker_alive"] else None

    return wait_for(check)


def test_authenticated_product_development_transfer_and_publication(api_client, service):
    client, _, state = api_client
    url = "/api/chat/sessions/web-owner/science-goals"
    lookup = url + "/generated-steps?case=recipient_sum"
    assert client.get(lookup).status_code == 401
    client.headers["x-session-access-token"] = "valid-test-token"
    assert client.get(lookup).json() == {"entries": []}
    catalog_response = client.get("/api/chat/science-goals/catalog")
    assert catalog_response.status_code == 200
    details = catalog_response.json()
    assert details["available"]
    assert {x["id"] for x in details["drivers"]} == {"paper_step.v1", "paper_step_transfer.v1"}
    created = client.post(url, json={**request(), "request_id": "web-source-code-01"})
    assert created.status_code == 200, created.text
    original = complete(service, "web-owner", created.json()["id"])
    assert original["code_step"]["registered"] and original["code_step"]["published"]
    assert original["usage"]["model_calls"] == 2 and original["code_step"]["completed_inputs"] == 1
    assert original["code_step"]["reused"] and state["chat_mode"] == "science_expert"
    found = client.get(lookup)
    assert found.headers["cache-control"] == "no-store"
    entry = found.json()["entries"][0]
    assert entry["origin_record_kind"] == "simulated_model_records"
    assert len(service.generated_steps("another-owner", "recipient_sum")) == 0
    assert service.generated_steps("web-owner", "source_sum") == []
    transferred = client.post(url, json={**request(True, entry["entry_id"]), "request_id": "web-import-code-01"})
    assert transferred.status_code == 200, transferred.text
    final = complete(service, "web-owner", transferred.json()["id"])
    assert final["code_step"]["imported"] and final["code_step"]["published"] and final["code_step"]["reused"]
    assert final["code_step"]["entry_id"] == original["code_step"]["entry_id"]
    assert final["usage"]["model_calls"] == 1 and final["usage"]["tool_calls"] == 3
    public_response = {"catalog": details, "goal": final, "entries": found.json()}

    def check_public_keys(value):
        if isinstance(value, dict):
            assert not {"references", "source_code", "credential"}.intersection(value)
            for item in value.values():
                check_public_keys(item)
        elif isinstance(value, list):
            for item in value:
                check_public_keys(item)

    check_public_keys(public_response)
    assert final["cycles"][0]["phase"] == "import"
    serialized = json.dumps(public_response)
    for private in [
        "PRIVATE_REFERENCE_SENTINEL",
        "unused-synthetic-provider-key",
        "source_code",
        str(service.state_root),
        str(service.root),
    ]:
        assert private not in serialized
    assert len(state["history"]) == 2


def test_publication_can_retry_without_repeating_computation(service):
    started = service.create("owner", request(), "publication-retry-01")
    complete(service, "owner", started["id"])
    folder, _ = service.paths("owner", started["id"])
    before = GoalStore(folder).read()
    (folder / "step_publication.json").unlink()
    assert not service.public("owner", started["id"])["code_step"]["published"]
    service.control("owner", started["id"], "publish")
    final = complete(service, "owner", started["id"])
    assert final["code_step"]["published"] and GoalStore(folder).read() == before
    assert len(list((folder.parent / ".generated_steps/generated_entries").glob("*.json"))) == 1


@pytest.mark.parametrize("change", ["path", "entry", "budget"])
def test_untrusted_parameters_rejected_before_preparation(service, change):
    spec = request()
    if change == "path":
        spec["parameters"]["definition"] = "/etc/passwd"
    elif change == "entry":
        spec = request(True, "../../another-owner")
    else:
        spec["budget"]["max_cycles"] = 1
    with pytest.raises(ValueError):
        service.create("owner", spec, "rejected-config-01")
    assert not service.children


def test_catalog_definition_and_symlink_drift_are_rejected(service, tmp_path):
    path = service.root / "configs/rsi/paper_step_definitions/source_sum.json"
    original = path.read_text()
    path.write_text(original + " ")
    with pytest.raises(ValueError, match="changed"):
        catalog(service.root)
    path.write_text(original)
    alias = tmp_path / "outside.json"
    alias.write_text(original)
    path.unlink()
    path.symlink_to(alias)
    with pytest.raises(ValueError):
        catalog(service.root)


def test_public_progress_rejects_changed_registration_receipt(service):
    started = service.create("owner", request(), "changed-receipt-01")
    final = complete(service, "owner", started["id"])
    folder, _ = service.paths("owner", started["id"])
    identity = final["code_step"]["capability_id"]
    p = folder / "host_private/registered" / f"{identity}.json"
    p.chmod(0o644)
    value = json.loads(p.read_text())
    value["payload"]["source_sha256"] = "f" * 64
    atomic_json(p, value)
    assert public_step(folder, GoalStore(folder).read()) == {"evidence_available": False}
    with pytest.raises(GoalConflict):
        service.control("owner", started["id"], "resume")


def test_completed_request_retry_survives_catalog_change(service):
    original = request()
    created = service.create("stable-owner", original, "stable-source-01")
    complete(service, "stable-owner", created["id"])
    before = dict(service.children)
    atomic_json(service.root / "configs/rsi/paper_steps.json", {"schema": 1, "tasks": {}})
    replay = service.create("stable-owner", original, "stable-source-01")
    assert replay["id"] == created["id"] and replay["status"] == "completed"
    assert service.children == before
