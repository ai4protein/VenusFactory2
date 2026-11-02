import hashlib
import json

import pytest

from agent.harness.contracts import digest, file_hash
from agent.harness.host import write_json
from agent.rsi import training_attempt_memory as memory


def episode(root, index=0, *, sources=None, feedback=None, reverse_results=False):
    folder = root / "episodes" / f"cycle_{index:03d}"
    work, execution = folder / "workspace", folder / "execution"
    cap = folder / "capability/bundles/versions/version"
    for p in (work, execution, cap):
        p.mkdir(parents=True)
    training = [{"id": "train", "sequence": "AC", "target": 1.0}]
    scope = {"protein": "fixture"}
    protocol = {"training_attempt_memory": memory.POLICY, "scope": scope, "data_hashes": {"training": digest(training)}}
    write_json(work / "training.json", training)
    write_json(cap / "task.json", {"phase": "develop", "scope": scope, "inputs": {"training.json": digest(training)}})
    lock = {
        "profile": {"budget": {"tool_calls": 12}},
        "task": {
            "phase": "develop",
            "tools": ["evaluate_model_code"],
            "inputs": {"training.json": file_hash(work / "training.json")},
            "capabilities": {"task.json": file_hash(cap / "task.json")},
        },
    }
    lock["sha256"] = digest(lock)
    write_json(execution / "profile.lock.json", lock)
    write_json(execution / "run.json", {"status": "incomplete", "profile_sha256": lock["sha256"]})
    sources = sources or ["def fit(): pass", "def predict(): pass"]
    feedback = feedback or {
        "status": "failed",
        "stage": "source_validation",
        "error_type": "ValueError",
        "message": "Malformed reference declaration",
    }
    starts, ends = [], []
    for i, source in enumerate(sources, 1):
        rationale = "training repair " + str(i)
        starts.append(
            {
                "kind": "tool_call_start",
                "tool_name": "mcp__venusfactory__evaluate_model_code",
                "tool_call_id": str(i),
                "args": {"source_code": source, "rationale": rationale},
            }
        )
        ends.append(
            {
                "kind": "tool_result",
                "tool_call_id": str(i),
                "is_error": False,
                "output": json.dumps(
                    {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "training_only_feedback": feedback}
                ),
            }
        )
        write_json(
            work / f"code_attempt_{i}.json",
            {"source_code": source, "rationale": rationale, "training_feedback": feedback},
        )
    events = starts + (list(reversed(ends)) if reverse_results else ends)
    (execution / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
    receipt = memory.capture(folder, protocol)
    write_json(folder / "attempt_evidence.json", receipt)
    cycle = {"index": index, "phase": "develop", "status": "failed", "attempt_evidence_sha256": digest(receipt)}
    return protocol, cycle, folder


def test_paired_training_errors_and_code_survive_a_new_episode(tmp_path):
    protocol, cycle, folder = episode(tmp_path, reverse_results=True)
    # Decoy independent data must never be opened or projected.
    write_json(folder / "acceptance.json", {"secret_label": "ASSESSMENT_SENTINEL"})
    write_json(tmp_path / "future_metrics.json", {"secret_label": "FUTURE_SENTINEL"})
    packet = memory.project(tmp_path, {"cycles": [cycle]}, protocol, "develop")
    assert len(packet["entries"]) == 2
    assert packet["entries"][0]["source_code"] == "def fit(): pass"
    assert packet["entries"][0]["training_only_feedback"]["message"] == "Malformed reference declaration"
    assert "SENTINEL" not in json.dumps(packet)
    assert "TRAINING ATTEMPT HISTORY" in memory.context(packet)
    assert not memory.project(tmp_path, {"cycles": [cycle]}, protocol, "reuse")["entries"]


@pytest.mark.parametrize("tamper", ["source", "feedback", "event", "training", "scope", "receipt", "symlink"])
def test_modified_training_history_fails_closed(tmp_path, tamper):
    protocol, cycle, folder = episode(tmp_path)
    p = folder / "workspace/code_attempt_1.json"
    if tamper in {"source", "feedback"}:
        value = json.loads(p.read_text())
        value["source_code" if tamper == "source" else "training_feedback"] = "changed"
        write_json(p, value)
    elif tamper == "event":
        p = folder / "execution/events.jsonl"
        p.write_text(p.read_text().replace("Malformed reference", "Forged reference"))
    elif tamper == "training":
        write_json(folder / "workspace/training.json", [])
    elif tamper == "scope":
        protocol["scope"] = {"protein": "another"}
    elif tamper == "receipt":
        cycle["attempt_evidence_sha256"] = "0" * 64
    else:
        destination = folder / "outside.json"
        p.rename(destination)
        p.symlink_to(destination)
    with pytest.raises(ValueError):
        memory.project(tmp_path, {"cycles": [cycle]}, protocol, "develop")


def test_metrics_projection_excludes_predictions_and_unknown_fields(tmp_path):
    protocol, cycle, _ = episode(
        tmp_path,
        feedback={"rows": 20, "rmse": 1.1, "predictions_sha256": "fingerprint", "future_score": 42, "weights": [99]},
    )
    entries = memory.project(tmp_path, {"cycles": [cycle]}, protocol, "develop")["entries"]
    assert all(e["training_only_feedback"] == {"rows": 20, "rmse": 1.1} for e in entries)


def test_history_is_bounded_and_large_source_is_not_silently_truncated(tmp_path):
    cycles = []
    for index in range(3):
        protocol, cycle, _ = episode(tmp_path, index, sources=["x" * 12001])
        cycles.append(cycle)
    entries = memory.project(tmp_path, {"cycles": cycles}, protocol, "develop")["entries"]
    assert [e["cycle"] for e in entries] == [1, 2]
    assert all("source_code" not in e and "source_omitted" in e for e in entries)


def test_legacy_and_source_withheld_policies_are_preserved(tmp_path):
    assert memory.project(tmp_path, {"cycles": [{"phase": "develop"}]}, {}, "develop")["entries"] == []
    for history in ["metrics_only", "independent_restart"]:
        with pytest.raises(ValueError, match="Source-withheld"):
            memory.policy_for({"training_attempt_memory": memory.POLICY, "history_policy": history})
    with pytest.raises(ValueError, match="Unknown"):
        memory.policy_for({"training_attempt_memory": "arbitrary"})


def test_unpaired_trace_is_unavailable_not_a_repair_authority(tmp_path):
    protocol, _, folder = episode(tmp_path)
    p = folder / "execution/events.jsonl"
    p.write_text("\n".join(p.read_text().splitlines()[:-1]) + "\n")
    receipt = memory.capture(folder, protocol)
    assert receipt["status"] == "unavailable" and receipt["attempts"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", [memory.POLICY, memory.CUMULATIVE_POLICY])
async def test_goal_supplies_failed_code_and_diagnostics_to_next_episode(tmp_path, monkeypatch, policy):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rsi"))
    import run_science_goal as base
    from test_reap_science_goal import setup_goal

    from agent.harness import Profile

    store = setup_goal(tmp_path)
    protocol = json.loads((store.root / "protocol.json").read_text())
    protocol["training_attempt_memory"] = policy
    write_json(store.root / "protocol.json", protocol)
    if policy == memory.CUMULATIVE_POLICY:

        def bind(state):
            state["definition"][memory.FIELD] = policy
            state["definition_sha256"] = digest(state["definition"])

        store.mutate(bind)
    profile = Profile.from_dict(store.read()["definition"]["profile"])
    seen = []

    async def execute(profile, task, runtimes, credential):
        seen.append(task.prompt)
        task.output.mkdir()
        lock = {"profile": profile.public(), "task": task.binding()}
        lock["sha256"] = digest(lock)
        write_json(task.output / "profile.lock.json", lock)
        source = "def fit(): pass"
        feedback = {
            "status": "failed",
            "stage": "source_validation",
            "error_type": "ValueError",
            "message": "Correct the fit/predict ABI",
        }
        write_json(
            task.workspace / "code_attempt_1.json",
            {"source_code": source, "rationale": "test", "training_feedback": feedback},
        )
        events = [
            {
                "kind": "tool_call_start",
                "tool_name": "mcp__venusfactory__evaluate_model_code",
                "tool_call_id": "one",
                "args": {"source_code": source, "rationale": "test"},
            },
            {
                "kind": "tool_result",
                "tool_call_id": "one",
                "is_error": False,
                "output": json.dumps(
                    {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "training_only_feedback": feedback}
                ),
            },
        ]
        if policy == memory.CUMULATIVE_POLICY and len(seen) in (2, 3):
            events = []
        (task.output / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events) + ("\n" if events else ""))
        run = {
            "status": "incomplete",
            "session_id": "fixture-" + str(len(seen)),
            "profile_sha256": lock["sha256"],
            "host_wall_seconds": 0.1,
            "audit": {"requests": {"model_calls": 1}, "events": {"tool_calls": 1}},
        }
        write_json(task.output / "run.json", run)
        return run

    monkeypatch.setattr(base, "execute", execute)
    monkeypatch.setattr(base, "audit_events", lambda *args: {"passed": False})
    goal = base.GeneratedModelGoal(store, "fixture")
    state = store.read()
    count = 4 if policy == memory.CUMULATIVE_POLICY else 2
    for i in range(count):
        outcome = await goal.cycle(i, profile, state, "Original objective")
        assert outcome["status"] == "failed"
        cycle = {"index": i, **outcome}
        memory.verify_episode(store.root, state, protocol, cycle)
        state["cycles"].append(cycle)
    assert "Correct the fit/predict ABI" not in seen[0]
    assert "Correct the fit/predict ABI" in seen[-1] and "def fit(): pass" in seen[-1]
    if policy == memory.CUMULATIVE_POLICY:
        assert '"empty_episodes": 2' in seen[-1] and "CUMULATIVE TRAINING ATTEMPT ARCHIVE" in seen[-1]
    else:
        assert "TRAINING ATTEMPT HISTORY" in seen[-1]
    packet_path = store.root / f"episodes/cycle_{count - 1:03d}/attempt_memory.json"
    packet = json.loads(packet_path.read_text())
    packet["entries"][0]["source_code"] = "altered"
    write_json(packet_path, packet)
    with pytest.raises(ValueError, match="differs"):
        memory.verify_episode(store.root, {"cycles": state["cycles"][:-1]}, protocol, cycle)
