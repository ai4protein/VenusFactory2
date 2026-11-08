import hashlib
import json

import pytest

from agent.harness.research_workspace import invoke_bundle


@pytest.mark.asyncio
@pytest.mark.parametrize("source_path", ["predict.py", "model/predict.py"])
@pytest.mark.parametrize("use_dataclass", [True, False])
async def test_delivered_python_classes_keep_import_identity_and_relative_weights(tmp_path, source_path, use_dataclass):
    bundle = tmp_path/"bundle"
    source = bundle/source_path
    source.parent.mkdir(parents=True)
    cls = ("from dataclasses import dataclass\n@dataclass\nclass Estimate:\n    value: float\n"
           if use_dataclass else
           "class Estimate:\n    def __init__(self, value): self.value = value\n")
    source.write_text(
        "from __future__ import annotations\n"
        "import json,pickle\nfrom pathlib import Path\n" + cls +
        "def run(inputs):\n"
        "    weight=json.loads(Path(__file__).with_name('weights.json').read_text())['scale']\n"
        "    result=pickle.loads(pickle.dumps(Estimate(inputs['value']*weight)))\n"
        "    return {'prediction':result.value}\n")
    weights = source.with_name("weights.json")
    weights.write_text(json.dumps({"scale":3.0}))
    paths = [source,weights]
    manifest = {"source":source_path,
                "files":{p.relative_to(bundle).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    result = await invoke_bundle(bundle, manifest, [{"value":2.0},{"value":-1.0}], tmp_path/"calls", batch=True)
    assert result == [{"prediction":6.0},{"prediction":-3.0}]
    # A fresh sandbox must import the same module and reopen the packaged weights.
    assert await invoke_bundle(bundle, manifest, {"value":4.0}, tmp_path/"again") == {"prediction":12.0}
