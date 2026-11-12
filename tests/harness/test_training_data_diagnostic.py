"""Training-only data access feedback; fixtures are not scientific model results."""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from agent.harness.research_training_errors import training_data_diagnostic

PREPARED = {"preparation": {"source_file": "inputs/binding_ddg.csv"}}


def missing(path):
    return str(FileNotFoundError(2, "No such file or directory", path))


def failure(path):
    return "FileNotFoundError: " + missing(path)


@pytest.mark.parametrize("path", ["inputs/binding_ddg.csv", "/workspace/inputs/binding_ddg.csv",
                                  "./inputs/binding_ddg.csv"])
def test_only_current_prepared_table_gets_interface_diagnostic(path):
    result = training_data_diagnostic(failure(path), PREPARED, stage="fitting")
    assert result["code"] == "prepared_training_file_not_mounted"
    assert result["data_access"] == "training_argument"
    assert "fit(training, seed)" in result["hint"] and "training['train']" in result["hint"]
    assert "validation" not in result["hint"] and "test" not in result["hint"]


@pytest.mark.parametrize("path", ["weights.pkl", "inputs/other.csv", "inputs/binding_ddg.csv.extra",
                                  "/other/inputs/binding_ddg.csv", "../inputs/binding_ddg.csv",
                                  "/workspace/../inputs/binding_ddg.csv"])
def test_unrelated_missing_file_never_recommends_training_rows(path):
    assert training_data_diagnostic(failure(path), PREPARED, stage="fitting") is None


@pytest.mark.parametrize("details,data,stage", [
    (failure("inputs/binding_ddg.csv"), PREPARED, "evaluation"),
    (failure("inputs/binding_ddg.csv"), {}, "fitting"),
    (failure("inputs/binding_ddg.csv"), {"preparation": None}, "fitting"),
    ("KeyError: 'inputs/binding_ddg.csv'", PREPARED, "fitting"),
    ("PermissionError: [Errno 13] Permission denied: 'inputs/binding_ddg.csv'", PREPARED, "fitting"),
    (failure("inputs/binding_ddg.csv") + "\nRuntimeError: another final cause", PREPARED, "fitting"),
    ("FileNotFoundError: [Errno 2] No such file or directory: not_a_string", PREPARED, "fitting"),
])
def test_no_diagnostic_when_context_or_final_cause_does_not_match(details, data, stage):
    assert training_data_diagnostic(details, data, stage=stage) is None


def test_observed_traceback_retains_cause_and_yields_bounded_contract_hint():
    fixture = json.loads((Path(__file__).parent / "fixtures/training_file_not_mounted.json").read_text())
    diagnostic = training_data_diagnostic(fixture["stderr"], PREPARED, stage="fitting")
    assert diagnostic["code"] == "prepared_training_file_not_mounted"
    assert len(json.dumps(diagnostic)) < 400
    assert "/home/" not in json.dumps(fixture)


async def prepared_service(tmp_path):
    import hashlib

    from agent.harness.research_data_store import ResearchDataStore
    from agent.harness.research_workspace import ResearchWorkspace
    work = tmp_path / "workspace"
    (work / "inputs").mkdir(parents=True)
    source = "inputs/binding_ddg.csv"
    raw = ("id,group,x,target\n" + "".join(f"r{i},g{i},{i},{2*i+1}\n" for i in range(36))).encode()
    (work / source).write_bytes(raw)
    store = ResearchDataStore(tmp_path / "private", work)
    store.protect_inputs([{"path": source, "name": "binding_ddg.csv", "sha256": hashlib.sha256(raw).hexdigest()}])
    service = ResearchWorkspace(work, tmp_path / "control", {"steps": []}, data_store=store)
    prepared = await service.prepare_research_dataset(data_file=source, target_column="target", input_columns=["x"],
        group_column="group", id_column="id", provenance="Synthetic software regression only",
        grouping="Disjoint fixture groups", seed=7)
    return service, prepared


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["inputs/binding_ddg.csv", "/workspace/inputs/binding_ddg.csv"])
async def test_real_isolated_fit_returns_diagnostic_to_native_mcp_caller(tmp_path, monkeypatch, path):
    from agent.harness import web_native_worker as worker
    from agent.harness.contracts import Budget, ModelBinding, Profile, digest
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("VENUS_SANDBOX_CONTROL_ROOT", str(tmp_path / "sandbox-slots"))
    service, prepared = await prepared_service(tmp_path)
    (tmp_path / "native-control").mkdir()
    # Authored fixture intentionally reads the outer table at module import,
    # exactly where the observed attempt failed. The fit interface is valid.
    code = (f"from pathlib import Path\nPath({path!r}).read_text()\n"
            "def fit(training, seed):\n"
            "    rows = training['train']\n"
            "    return {'mean': sum(row['target'] for row in rows) / len(rows)}\n"
            "def predict(model, inputs):\n"
            "    return {'value': model['mean']}\n")
    await service.write_research_file("algorithm.py", code)
    messages = []

    async def adapter(ctx):
        config = json.loads((ctx.task.capability / "connection.json").read_text())
        async with httpx.AsyncClient(trust_env=False, timeout=60) as client:
            async def call(name, arguments):
                response = await client.post(config["url"], headers={"Authorization": "Bearer " + config["token"]},
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                          "params": {"name": name, "arguments": arguments}})
                response.raise_for_status()
                return response.json()["result"]
            await call("describe_tool", {"name": "train_research_model"})
            result = await call("call_tool", {"name": "train_research_model", "arguments": {
                "name": "broken fixture", "source_file": "algorithm.py", "evaluation_file": prepared["evaluation_file"],
                "hypothesis": "Exercise isolated fitting error feedback"}})
            assert result["isError"]
            text = result["content"][0]["text"]
            messages.append(text)
            assert "FileNotFoundError" in text and "TrainingAttemptError" in text
            assert '"code": "prepared_training_file_not_mounted"' in text
            assert '"data_access": "training_argument"' in text and "training['train']" in text
            assert str(tmp_path) not in text
        return {"status": "completed", "text": "Offline error delivery verified"}

    monkeypatch.setattr(worker, "builtin_registry", lambda: SimpleNamespace(
        resolve=lambda *args: (SimpleNamespace(run=adapter), None)))
    profile = Profile("science_agent", "kimi-code", ModelBinding("fixture", "unused", "http://127.0.0.1:1"),
                      Budget(tool_calls=1))
    await worker.run({"profile": profile.public(), "workspace": str(service.workspace),
        "control": str(tmp_path / "native-control"), "session_id": "training-diagnostic-fixture",
        "runtimes": {}, "base_url": "http://127.0.0.1:1", "token": "fixture", "text": "Offline contract test"},
        tools=service.tools(), event_sink=lambda event: None)
    records = [json.loads(p.read_text()) for p in (service.development.root / "training").glob("*/receipt.json")]
    assert len(records) == 1 and len(messages) == 1
    receipt = records[0]
    assert receipt["status"] == "failed" and receipt["stage"] == "fitting"
    assert receipt["execution"]["exit_code"] != 0
    assert receipt["execution"]["backend"] == "bwrap"
    assert receipt["diagnostic"]["data_access"] == "training_argument"
    assert receipt["receipt_sha256"] == digest({k: v for k, v in receipt.items() if k != "receipt_sha256"})
    assert not receipt["validation_and_test_mounted"] and not service.development.records()
    assert (service.workspace / "algorithm.py").read_text() == code


@pytest.mark.asyncio
@pytest.mark.parametrize("stage,path", [("fitting", "weights.pkl"), ("evaluation", "inputs/binding_ddg.csv")])
async def test_real_unrelated_or_prediction_error_keeps_original_meaning(tmp_path, monkeypatch, stage, path):
    from agent.harness.research_attempt_resolution import TrainingAttemptError
    monkeypatch.setenv("VENUS_JOB_QUEUE_ROOT", str(tmp_path / "queue"))
    monkeypatch.setenv("VENUS_SANDBOX_CONTROL_ROOT", str(tmp_path / "sandbox-slots"))
    service, prepared = await prepared_service(tmp_path)
    read = f"    Path({path!r}).read_text()\n"
    code = ("from pathlib import Path\ndef fit(training, seed):\n" + (read if stage == "fitting" else "") +
            "    rows = training['train']\n"
            "    return {'mean': sum(row['target'] for row in rows) / len(rows)}\n"
            "def predict(model, inputs):\n" + (read if stage == "evaluation" else "") +
            "    return {'value': model['mean']}\n")
    await service.write_research_file("algorithm.py", code)
    with pytest.raises(TrainingAttemptError) as captured:
        await service.train_research_model("negative fixture", "algorithm.py", prepared["evaluation_file"],
            "Preserve unrelated and prediction errors")
    assert "FileNotFoundError" in str(captured.value) and path in str(captured.value)
    assert "prepared_training_file_not_mounted" not in str(captured.value)
    receipt = json.loads(next((service.development.root / "training").glob("*/receipt.json")).read_text())
    assert receipt["stage"] == stage and "diagnostic" not in receipt
    assert not service.development.records()
