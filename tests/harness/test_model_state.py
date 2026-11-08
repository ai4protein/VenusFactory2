"""Real sandbox persistence fixtures; these do not count as biological models."""
import hashlib
import json
import pickle
from pathlib import Path

import pytest

from agent.harness.research_workspace import ResearchWorkspace, run_python
from agent.harness.capability_review import model_evaluation
from agent.harness.research_model_state import read_fitted_state, validate_manifest


def dataset():
    def rows(start, count, label):
        return [{"id":label+str(i), "group":label+str(i),
                 "inputs":{"x":i}, "target":2*i+1} for i in range(start,start+count)]
    return {"task":"regression", "provenance":"Synthetic software fixture",
            "grouping":"Independent groups for protocol checks",
            "splits":{"train":rows(1,8,"t"),"validation":rows(9,3,"v"),"test":rows(12,3,"q")}}


async def service_at(tmp_path, source):
    service=ResearchWorkspace(tmp_path/"workspace",tmp_path/"control",{"steps":[]},
                              development_root=tmp_path/"development")
    await service.write_research_file("evaluation.json",json.dumps(dataset()))
    await service.write_research_file("algorithm.py",source)
    return service


SOURCES = {
    "sklearn": """from sklearn.linear_model import LinearRegression
def fit(training, seed):
    rows=training["train"]
    return LinearRegression().fit([[r["inputs"]["x"]] for r in rows],[r["target"] for r in rows])
def predict(model, inputs):
    return {"value":float(model.predict([[inputs["x"]]])[0])}
""",
    "arrays": """import numpy as np
def fit(training, seed):
    rows=training["train"]
    coefficients=np.linalg.lstsq(np.array([[1,r["inputs"]["x"]] for r in rows]),np.array([r["target"] for r in rows]),rcond=None)[0]
    return {"coefficients":coefficients}
def predict(model, inputs):
    return {"value":float(model["coefficients"] @ np.array([1,inputs["x"]]))}
""",
    "custom": """class Predictor:
    def __init__(self, slope, offset):
        self.slope,self.offset=slope,offset
    def predict(self, x):
        return self.slope*x+self.offset
def fit(training, seed):
    rows=training["train"]
    a,b=rows[0],rows[-1]
    slope=(b["target"]-a["target"])/(b["inputs"]["x"]-a["inputs"]["x"])
    return Predictor(slope,a["target"]-slope*a["inputs"]["x"])
def predict(model, inputs):
    return {"value":model.predict(inputs["x"])}
"""
}


@pytest.mark.asyncio
@pytest.mark.parametrize("implementation", list(SOURCES))
async def test_estimator_arrays_and_custom_state_fit_replay_and_refit(tmp_path, monkeypatch, implementation):
    service=await service_at(tmp_path,SOURCES[implementation])
    def host_unpickle_forbidden(*args,**kwargs):
        raise AssertionError("Opaque state was deserialized outside the sandbox")
    monkeypatch.setattr(pickle,"load",host_unpickle_forbidden)
    monkeypatch.setattr(pickle,"loads",host_unpickle_forbidden)
    result=await service.train_research_model("fixture","algorithm.py","evaluation.json","Protocol fixture")
    receipt=result["training"]; experiment=result["experiment"]
    assert receipt["state_format"]=="pickle-v5"
    assert experiment["validation"]["score"]==pytest.approx(0,abs=1e-10)
    assert experiment["fresh_process_replay"]
    assert receipt["validation_and_test_mounted"] is False
    if implementation=="sklearn":
        assert {"numpy","scikit-learn"} <= set(receipt["dependency_versions"])
    if implementation=="arrays":
        assert "numpy" in receipt["dependency_versions"]
    runtime=service.development.root/"experiments"/experiment["id"]/"runtime"
    assert not list(runtime.rglob("training.json"))
    spec={"input_schema":{"type":"object","properties":{"x":{"type":"number"}},"required":["x"]},
          "output_schema":{"type":"object","properties":{"value":{"type":"number"}},"required":["value"]},
          "evaluation_file":receipt["evaluation_file"],"prediction_key":"value",
          "reproduction_script":receipt["reproduction_script"],"fitted_files":receipt["fitted_files"]}
    manifest={"source":receipt["source"],"files":experiment["files"],"review_spec":spec,
              "evidence_files":{name:"private" for name in receipt["evidence_files"]}}
    review=await model_evaluation(service.workspace,manifest,tmp_path/"review",run_python)
    assert review["training_reproduced"] and review["score"]==pytest.approx(0,abs=1e-10)


@pytest.mark.asyncio
async def test_pickle_reconstruction_cannot_write_outside_mounted_workspace(tmp_path):
    sentinel=tmp_path/"host-only"/"tripwire.txt"
    sentinel.parent.mkdir()
    source="""from pathlib import Path
def reconstruct():
    try:
        Path(REPLACE_PATH).write_text("escaped")
    except OSError:
        return 1.0
    raise RuntimeError("Sandbox isolation failed")
class Probe:
    def __reduce__(self):
        return reconstruct, ()
def fit(training, seed):
    return {"probe":Probe(),"mean":sum(r["target"] for r in training["train"])/len(training["train"])}
def predict(model, inputs):
    return {"value":model["mean"]+model["probe"]-1}
""".replace("REPLACE_PATH",repr(str(sentinel)))
    service=await service_at(tmp_path,source)
    result=await service.train_research_model("isolation","algorithm.py","evaluation.json","Controlled isolation check")
    assert result["experiment"]["fresh_process_replay"]
    assert not sentinel.exists()


@pytest.mark.parametrize("field,value",[
    ("format", "joblib"),("file","../outside.pkl"),("file","/tmp/model.pkl"),
    ("bytes",67108865),("sha256","not-a-hash"),("dependencies",{"x;touch nope":"1"})
])
def test_invalid_transport_rejected_without_deserialization(tmp_path, field, value):
    raw=b"opaque and deliberately not a pickle"
    metadata={"schema":"venusfactory.fitted-model.v1","format":"pickle-v5","file":"model.pkl",
              "bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"python":"3.12","dependencies":{}}
    metadata[field]=value
    (tmp_path/"model-state.json").write_text(json.dumps(metadata))
    (tmp_path/"model.pkl").write_bytes(raw)
    with pytest.raises(ValueError):
        read_fitted_state(tmp_path)


def test_opaque_bytes_only_hashed_on_host_and_tampering_rejected(tmp_path):
    raw=b"opaque and deliberately not a pickle"
    metadata={"schema":"venusfactory.fitted-model.v1","format":"pickle-v5","file":"model.pkl",
              "bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"python":"3.12","dependencies":{}}
    (tmp_path/"model-state.json").write_text(json.dumps(metadata))
    (tmp_path/"model.pkl").write_bytes(raw)
    value,files,digest=read_fitted_state(tmp_path)
    assert files["model.pkl"]==raw and digest==metadata["sha256"]
    (tmp_path/"model.pkl").write_bytes(b"tampered")
    with pytest.raises(ValueError,match="does not match"):
        read_fitted_state(tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport",["estimator","large_json"])
async def test_review_private_delivery_and_agent_reuse(tmp_path,transport):
    import zipfile
    from agent.harness.capability_review import review_proposal
    from agent.harness.research_development import build_delivery
    from agent.harness.research_workspace import publish_proposals, run_capability

    baseline="""def fit(training, seed):
    return {"mean":sum(r["target"] for r in training["train"])/len(training["train"])}
def predict(model, inputs):
    return {"value":model["mean"]}
"""
    service=await service_at(tmp_path,baseline)
    first=await service.train_research_model("mean","algorithm.py","evaluation.json","Controlled mean reference")
    code=SOURCES["sklearn"] if transport=="estimator" else """def fit(training, seed):
    a,b=training["train"][0],training["train"][-1]
    slope=(b["target"]-a["target"])/(b["inputs"]["x"]-a["inputs"]["x"])
    return {"slope":slope,"offset":a["target"]-slope*a["inputs"]["x"],"padding":[0.0]*9000}
def predict(model, inputs):
    if type(inputs.get("x")) not in (int,float):raise ValueError("Numeric x required")
    return {"value":model["slope"]*inputs["x"]+model["offset"]}
"""
    await service.write_research_file("algorithm.py",code)
    result=await service.train_research_model("linear","algorithm.py","","Controlled linear model",parent_experiment_id=first["experiment"]["id"])
    receipt=result["training"];experiment=result["experiment"]
    await service.write_research_file("README.md","Synthetic linear-response fixture. Not protein data, biological validation or algorithmic novelty.")
    spec={"kind":"model","author":"Software fixture","version":"1.0",
          "scope":{"proteins":["none: synthetic fixture"],"properties":["linear response"],
                   "scenarios":["software testing"],"limitations":["No biological claim or novelty"]},
          "input_schema":{"type":"object","properties":{"x":{"type":"number"}},"required":["x"],"additionalProperties":False},
          "output_schema":{"type":"object","properties":{"value":{"type":"number"}},"required":["value"],"additionalProperties":False},
          "dependencies":[],"cases":[{"inputs":{"x":2},"expected":{"value":5}},{"inputs":{},"reject":True}],
          "evidence_files":["README.md",*receipt["evidence_files"]],"evaluation_file":receipt["evaluation_file"],
          "prediction_key":"value","reproduction_script":receipt["reproduction_script"],"fitted_files":receipt["fitted_files"],
          "development_experiment_id":experiment["id"]}
    await service.propose_research_capability("Fixture","Controlled training persistence fixture",receipt["source"],receipt["fitted_files"],[{"x":1.5}],spec)
    bundle=service.proposals[0]
    async def reviewer(context):
        if transport=="estimator":
            assert "scikit-learn" in context["host_evidence"]["dependency_versions"]
        return {"decision":"approve","summary":"Synthetic software fixture only.",
                "checks":{k:{"passed":True,"reason":"Controlled nonbiological test"} for k in context["required_checks"]},
                "issues":[],"tests":[{"inputs":{"x":3},"expected":{"value":7}},{"inputs":{"x":"bad"},"reject":True}]}
    review=await review_proposal(bundle,reviewer=reviewer,control=tmp_path/"independent",runner=run_python)
    assert review["status"]=="approved",review
    delivery=build_delivery(service.workspace,bundle,review)
    with zipfile.ZipFile(service.workspace/delivery["model_package"]) as archive:
        assert not any(n.endswith(("training.json","evaluation.json")) for n in archive.namelist())
        if transport=="estimator":
            assert "scikit-learn==" in archive.read("requirements.txt").decode()
        hashes=json.loads(archive.read("checksums.json"))
        assert hashes
    library=tmp_path/"private-library"
    item=publish_proposals([bundle],"software-fixture",library)[0]
    output=await run_capability(item["id"],{"x":4.5},library=library)
    assert output["value"]==pytest.approx(10)
