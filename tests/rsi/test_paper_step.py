"""Host admission, isolated computation, immutable registration and fresh reuse."""

import json
from pathlib import Path

import pytest

from agent.rsi.paper_step import PaperStepLifecycle, compare, digest, validated_source
from agent.rsi.storage import sha256_file

SOURCE = "def reproduce(inputs):\n    return {'sum': sum(inputs)}\n"


def definition(tmp_path):
    ref = tmp_path / "reference.txt"
    ref.write_text("independent mathematical fixture")
    return {
        "schema": 1,
        "paper": {"title": "Numerical test fixture", "doi": "test:fixture"},
        "scope": {"covered_steps": ["summation"], "excluded_steps": ["scientific discovery"]},
        "interface": {"id": "sum.v1", "description": "numeric list -> sum object"},
        "references": {str(ref): sha256_file(ref)},
        "cases": {
            "development": [{"id": "dev", "inputs": [1, 2], "expected": {"sum": 3}}],
            "acceptance": [{"id": "accept", "inputs": [3, 5], "expected": {"sum": 8}}],
            "reuse": [{"id": "reuse", "inputs": [7, 11], "expected": {"sum": 18}}],
        },
        "budget": {"candidates": 2, "operation_seconds": 30},
        "tolerance": {"atol": 1e-12, "rtol": 0},
    }


@pytest.mark.parametrize(
    "code",
    [
        "def reproduce(x,y): return x",
        "def reproduce(x): pass",
        "def reproduce(x):\n return open('/etc/passwd').read()",
        "def reproduce(x): return x\ndef predict(m,x): return x",
    ],
)
def test_invalid_or_unavailable_abi_rejected(code):
    if code.endswith("pass"):
        # The execution adapter itself has substantive functions; the paper ABI
        # must reject placeholder reproduce functions as well.
        with pytest.raises(ValueError):
            validated_source(code)
    else:
        with pytest.raises(ValueError):
            validated_source(code)


def test_structure_bool_and_nonfinite_cannot_pass_as_numbers():
    assert compare({"a": [1, 2]}, {"a": [1.0, 2]}, atol=0, rtol=0)["passed"]
    assert not compare({"a": True}, {"a": 1}, atol=0, rtol=0)["passed"]
    assert not compare({"a": [1]}, {"a": [1, 2]}, atol=0, rtol=0)["passed"]
    assert not compare({"a": 1, "extra": 2}, {"a": 1}, atol=0, rtol=0)["passed"]
    with pytest.raises(ValueError):
        compare(float("nan"), 1, atol=0, rtol=0)


def test_admission_private_references_and_overlap_rejected(tmp_path):
    d = definition(tmp_path)
    lc = PaperStepLifecycle(tmp_path / "host", d)
    public = lc.public_task()
    assert public["acceptance_cases"] == 1 and "references" not in public
    assert "cases" not in public and "expected" not in public
    assert public["development_cases"] == d["cases"]["development"]
    d["cases"]["reuse"][0]["inputs"] = [3, 5]
    with pytest.raises(ValueError, match="distinct"):
        PaperStepLifecycle(tmp_path / "other", d)


def test_failed_source_consumes_budget_and_survives_restart(tmp_path):
    d = definition(tmp_path)
    d["budget"]["candidates"] = 1
    lc = PaperStepLifecycle(tmp_path / "host", d)
    with pytest.raises(ValueError):
        lc.propose("pass", declared_origin="test")
    lc = PaperStepLifecycle(tmp_path / "host", d)
    with pytest.raises(ValueError, match="budget"):
        lc.propose(SOURCE, declared_origin="test")
    assert (lc.root / "attempts/0001/failure.json").is_file()


def test_changed_reference_and_symlinks_fail(tmp_path):
    d = definition(tmp_path)
    lc = PaperStepLifecycle(tmp_path / "host", d)
    Path(next(iter(d["references"]))).write_text("changed")
    with pytest.raises(ValueError, match="reference changed"):
        lc.public_task()
    (tmp_path / "link").symlink_to(tmp_path / "host", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        PaperStepLifecycle(tmp_path / "link", d)


def test_real_sandbox_registration_restart_reuse_and_tampering(tmp_path):
    d = definition(tmp_path)
    lc = PaperStepLifecycle(tmp_path / "host", d)
    cid = lc.propose(SOURCE, declared_origin="controller_test_not_agent")
    with pytest.raises(FileNotFoundError):
        lc.register(cid)
    assert lc.assess(cid)["passed"]
    cap = lc.register(cid)
    assert lc.register(cid) == cap
    lc = PaperStepLifecycle(tmp_path / "host", d)
    assert lc.reuse(cap, "reuse")["passed"]
    assert lc.verify_reuse(cap, "reuse")["fresh_execution"]
    assert not lc.capability(cap)["agent_authorship_verified"]
    with pytest.raises(ValueError, match="already exists"):
        lc.reuse(cap, "reuse")
    path = lc.root / "assessments" / cid / "accept.json"
    obj = json.loads(path.read_text())
    obj["payload"]["output"]["sum"] = 0
    # Even rewriting the local envelope does not bypass execution/reference checks.
    obj["sha256"] = digest(obj["payload"])
    path.chmod(0o644)
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="execution evidence"):
        lc.capability(cap)


def test_wrong_computation_not_registered(tmp_path):
    d = definition(tmp_path)
    lc = PaperStepLifecycle(tmp_path / "host", d)
    cid = lc.propose("def reproduce(inputs):\n return {'sum': 3}\n", declared_origin="controller_test")
    assert not lc.assess(cid)["passed"]
    with pytest.raises(ValueError, match="did not pass"):
        lc.register(cid)


def test_hidden_host_file_is_not_mounted_and_failed_execution_cannot_restart(tmp_path):
    from agent.rsi.code_sandbox import CandidateCodeError

    d = definition(tmp_path)
    secret = tmp_path / "private_reference_values.txt"
    secret.write_text("8\n")
    lc = PaperStepLifecycle(tmp_path / "host", d)
    source = (
        "import numpy as np\ndef reproduce(inputs):\n    return {'sum': float(np.loadtxt(" + repr(str(secret)) + "))}\n"
    )
    cid = lc.propose(source, declared_origin="isolation_test")
    with pytest.raises(CandidateCodeError):
        lc.assess(cid)
    failure = lc._read("assessments/" + cid + "/failure.json")
    assert failure["candidate_failure"] and "FileNotFoundError" in failure["message"]
    lc = PaperStepLifecycle(tmp_path / "host", d)
    with pytest.raises(ValueError, match="already exists"):
        lc.assess(cid)
    with pytest.raises(FileNotFoundError):
        lc.register(cid)


def test_modified_admission_or_threshold_cannot_resume_existing_lifecycle(tmp_path):
    d = definition(tmp_path)
    PaperStepLifecycle(tmp_path / "host", d)
    d["tolerance"]["atol"] = 0.01
    with pytest.raises(ValueError, match="already exists"):
        PaperStepLifecycle(tmp_path / "host", d)
