"""Exercise downloaded releases without importing the platform or calling a model."""
import json
import os
import subprocess
import sys
import zipfile

import pytest

from agent.rsi.artifacts.package import build_delivery
from agent.harness.capability_review import review_proposal
from agent.harness.research_workspace import run_python
from test_research_development import measured, review_spec, independent


@pytest.mark.asyncio
async def test_downloaded_model_replays_developer_and_independent_invariants(tmp_path):
    service, baseline, candidate = await measured(tmp_path)
    await service.write_research_file("README.md", "Synthetic protocol fixture; no biological claim.")
    specification = review_spec(candidate["id"])
    specification["cases"].append({
        "inputs": {"x": 1.7},
        "invariants": {"finite_prediction": True, "min": 2.8, "max": 3.0},
    })
    async def reviewer(context):
        response = await independent(context)
        response["tests"][0] = {
            "inputs": {"x": 3},
            "invariants": {"finite_prediction": True, "min": 8.9, "max": 9.1},
        }
        return response
    await service.propose_research_capability(
        "Review replay fixture", "Synthetic nonlinear learner", "predict.py",
        ["weights.json"], [{"x": 1.5}], specification)
    bundle = service.proposals[0]
    receipt = await review_proposal(bundle, reviewer=reviewer, control=tmp_path/"review", runner=run_python)
    assert receipt["status"] == "approved", receipt
    delivery = build_delivery(service.workspace, bundle, receipt)
    destination = tmp_path/"standalone"
    with zipfile.ZipFile(service.workspace/delivery["package"]) as archive:
        archive.extractall(destination)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    result = subprocess.run(command, cwd=destination, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert json.loads((destination/"tests/independent-cases.json").read_text()) == receipt["independent"]["tests"]
    assert "test_independent_review" in result.stderr
    source = destination/"runtime/predict.py"
    original = source.read_text()
    # Only the independent review input is affected; the developer cases still pass.
    for replacement in (
        "if inputs.get('x') == 3: return {'value': 999}",
        "if inputs.get('x') == 3:\n        from pathlib import Path\n        p = Path('counter.txt')\n        n = int(p.read_text()) + 1 if p.exists() else 1\n        p.write_text(str(n))\n        return {'value': 9 + n * 0.01}",
    ):
        source.write_text(original.replace("def run(inputs):", "def run(inputs):\n    " + replacement))
        result = subprocess.run(command, cwd=destination, env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0, "Independent bounds and repeatability must be exercised"
        assert "test_independent_review" in result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["missing", "different_input", "wrong_count"])
async def test_delivery_requires_the_executed_independent_cases(tmp_path, change):
    from test_capability_admission import candidate, spec, reviewer
    from agent.harness.contracts import digest
    service = await candidate(tmp_path, spec())
    bundle = service.proposals[0]
    receipt = await review_proposal(bundle, reviewer=reviewer, control=tmp_path/"review", runner=run_python)
    assert receipt["status"] == "approved", receipt
    if change == "missing":
        receipt["independent"].pop("tests")
    elif change == "different_input":
        receipt["independent"]["tests"][0]["inputs"] = {"sequence": "TEST"}
    else:
        receipt["automatic"]["independent_cases"].pop()
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = digest(receipt)
    (bundle/"admission.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Independent review cases"):
        build_delivery(service.workspace, bundle, receipt)


@pytest.mark.asyncio
async def test_repeat_packaging_is_stable_and_new_packagers_preserve_old_releases(tmp_path, monkeypatch):
    import asyncio
    from agent.rsi.artifacts import package
    from test_capability_admission import candidate, spec, reviewer
    service = await candidate(tmp_path, spec())
    bundle = service.proposals[0]
    receipt = await review_proposal(bundle, reviewer=reviewer, control=tmp_path/"review", runner=run_python)
    assert receipt["status"] == "approved"
    first = build_delivery(service.workspace, bundle, receipt)
    original = (service.workspace/first["package"]).read_bytes()
    await asyncio.sleep(2.1)  # ZIP timestamps use a two-second resolution.
    repeated = build_delivery(service.workspace, bundle, receipt)
    assert repeated == first
    assert (service.workspace/first["package"]).read_bytes() == original
    monkeypatch.setattr(package, "RUNNER", package.RUNNER + "\n# Packaging revision fixture\n")
    revised = build_delivery(service.workspace, bundle, receipt)
    assert revised["candidate_id"] == first["candidate_id"]
    assert revised["package"] != first["package"]
    assert (service.workspace/first["package"]).read_bytes() == original
    changed = service.workspace/revised["package"]
    changed.write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="changed"):
        build_delivery(service.workspace, bundle, receipt)
    assert changed.read_bytes() == b"changed archive"
