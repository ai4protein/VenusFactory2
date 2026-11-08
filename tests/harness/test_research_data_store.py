import asyncio
import csv
import hashlib
import io
import json
from pathlib import Path

import pytest

from agent.harness.research_data_store import ResearchDataStore, SCHEMA
from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.research_input_index import input_descriptions
from agent.harness.research_data_memory import prepared_dataset_context


def setup_table(tmp_path, suffix=".csv"):
    work=tmp_path/"workspace";work.mkdir(parents=True)
    buffer=io.StringIO()
    writer=csv.DictWriter(buffer,fieldnames=["id","group","sequence","target"],
                          delimiter="\t" if suffix==".tsv" else ",")
    writer.writeheader()
    for i in range(36):
        writer.writerow({"id":"ROW_"+str(i),"group":"GROUP_"+str(i),
                         "sequence":"A"*(i+5)+"G"*(45-i),"target":9100000013+i*i*113+i*7})
    raw=buffer.getvalue().encode()
    path="inputs/measurements"+suffix
    (work/"inputs").mkdir()
    (work/path).write_bytes(raw)
    item={"path":path,"name":"measurements"+suffix,"sha256":hashlib.sha256(raw).hexdigest()}
    store=ResearchDataStore(tmp_path/"private",work)
    store.protect_inputs([item])
    service=ResearchWorkspace(work,tmp_path/"control",{"steps":[]},data_store=store)
    return work,store,service,item,raw


async def prepare(service, item, **kwargs):
    return await service.prepare_research_dataset(
        data_file=item["path"],target_column="target",input_columns=["sequence"],
        group_column="group",id_column="id",provenance="Synthetic isolation fixture",
        grouping="Distinct fixture groups",seed=7,**kwargs)


def hidden_markers(data):
    return [str(int(row["target"])) for key in ["validation","test"] for row in data["splits"][key]]


def workspace_text(work):
    return "\n".join(p.read_bytes().decode("utf-8",errors="ignore")
                     for p in work.rglob("*") if p.is_file())


@pytest.mark.parametrize("suffix",[".csv",".tsv"])
@pytest.mark.asyncio
async def test_agent_sees_only_schema_then_training_rows(tmp_path,suffix):
    work,store,service,item,original=setup_table(tmp_path,suffix)
    assert "9100000013" not in workspace_text(work)
    description=input_descriptions(work,[item],source_reader=store.input_bytes)
    assert {x["name"] for x in description["files"][0]["columns"]}=={"id","group","sequence","target"}
    assert "9100000013" not in json.dumps(description)
    result=await prepare(service,item)
    handle=json.loads((work/result["evaluation_file"]).read_text())
    assert handle["schema"]==SCHEMA and set(handle)=={"schema","id"}
    data=json.loads(store.load_evaluation(result["evaluation_file"]))
    assert store.original_bytes(item["path"])==original
    assert all(marker not in workspace_text(work) for marker in hidden_markers(data))
    assert all(marker not in json.dumps(result) for marker in hidden_markers(data))
    table=list(csv.DictReader(io.StringIO((work/item["path"]).read_text()),delimiter="\t" if suffix==".tsv" else ","))
    assert {r["id"] for r in table}=={r["id"] for r in data["splits"]["train"]}
    assert result["passed"]


@pytest.mark.asyncio
async def test_split_cannot_be_changed_to_reveal_heldout_rows(tmp_path):
    work,store,service,item,_=setup_table(tmp_path)
    result=await prepare(service,item)
    before=(work/item["path"]).read_bytes()
    with pytest.raises(ValueError,match="original held-out partition"):
        await service.prepare_research_dataset(data_file=item["path"],target_column="target",
            input_columns=["sequence"],group_column="group",id_column="id",
            provenance="fixture",grouping="fixture groups",seed=41,output_file="new.json")
    assert not (work/"new.json").exists()
    assert (work/item["path"]).read_bytes()==before


@pytest.mark.asyncio
async def test_restore_after_interruption_never_restores_raw_labels(tmp_path):
    work,store,service,item,raw=setup_table(tmp_path)
    (work/item["path"]).write_bytes(raw)
    restored=ResearchDataStore(tmp_path/"private",work)
    restored.protect_inputs([item],prior_agent_execution=True)
    assert "9100000013" not in workspace_text(work)
    result=await prepare(service,item)
    data=json.loads(store.load_evaluation(result["evaluation_file"]))
    (work/item["path"]).write_bytes(raw)
    restored.protect_inputs([item],prior_agent_execution=True)
    assert all(marker not in workspace_text(work) for marker in hidden_markers(data))


@pytest.mark.asyncio
async def test_metadata_resume_verifies_private_snapshot_and_training_view(tmp_path):
    work,store,service,item,_=setup_table(tmp_path)
    service.control=tmp_path/"episodes/cycle_000/tools";service.control.mkdir(parents=True)
    result=await prepare(service,item)
    resumed=prepared_dataset_context(work,tmp_path/"episodes",data_store=store)
    assert resumed["evaluation_sha256"]==result["evaluation_sha256"]
    assert resumed["columns"]["target_column"]=="target"
    assert "9100000013" not in json.dumps(resumed)
    handle=json.loads((work/result["evaluation_file"]).read_text())
    (store.root/"evaluations"/(handle["id"]+".json")).write_text("{}")
    assert prepared_dataset_context(work,tmp_path/"episodes",data_store=store) is None


@pytest.mark.asyncio
async def test_evaluation_references_are_goal_owned_and_fail_closed(tmp_path):
    work,store,service,item,_=setup_table(tmp_path/"first")
    result=await prepare(service,item)
    other_work=tmp_path/"other";other_work.mkdir()
    other=ResearchDataStore(tmp_path/"other-private",other_work)
    (other_work/"evaluation.json").write_bytes((work/result["evaluation_file"]).read_bytes())
    with pytest.raises((OSError,ValueError)):
        other.load_evaluation("evaluation.json")
    (work/"plain.json").write_text(json.dumps({"splits":{"test":[{"target":"HIDDEN"}]}}))
    with pytest.raises(ValueError,match="host-prepared"):
        store.load_evaluation("plain.json")
    for path in ["../private/evaluations/x.json","/etc/passwd"]:
        with pytest.raises((OSError,ValueError)):
            store.load_evaluation(path)
    (work/"linked.json").symlink_to(store.root/"inputs.json")
    with pytest.raises((OSError,ValueError)):
        store.load_evaluation("linked.json")


def test_unregistered_or_previously_exposed_tables_are_not_silently_blinded(tmp_path):
    work,store,service,item,raw=setup_table(tmp_path)
    alias="inputs/previous.csv";(work/alias).write_bytes(raw)
    with pytest.raises(ValueError,match="earlier execution"):
        store.protect_inputs([{**item,"path":alias}],prior_agent_execution=True)
    with pytest.raises(ValueError,match="original protected"):
        store.original_bytes(alias)
    with pytest.raises(ValueError,match="outside"):
        ResearchDataStore(work/"private",work)


def test_stale_store_instances_do_not_drop_concurrent_uploads(tmp_path):
    work,store,service,item,raw=setup_table(tmp_path)
    second=ResearchDataStore(store.root,work)
    store.ingest("inputs/second.csv",raw,item["sha256"])
    second.ingest("inputs/third.csv",raw,item["sha256"])
    actual=json.loads(store.manifest_path.read_text())
    assert {item["path"],"inputs/second.csv","inputs/third.csv"} <= set(actual)


@pytest.mark.asyncio
async def test_duplicate_upload_cannot_reveal_a_different_partition(tmp_path):
    work,store,service,item,raw=setup_table(tmp_path)
    await prepare(service,item)
    store.ingest("inputs/duplicate.csv",raw,item["sha256"])
    with pytest.raises(ValueError,match="original held-out partition"):
        await service.prepare_research_dataset(data_file="inputs/duplicate.csv",
            target_column="target",input_columns=["sequence"],group_column="group",
            id_column="id",provenance="fixture",grouping="fixture",seed=41)


@pytest.mark.asyncio
async def test_upload_staging_never_copies_measurements_to_workspace(tmp_path,monkeypatch):
    from backend.web_v2 import research_files
    monkeypatch.setenv("WEBUI_V2_MODE","local")
    work,store,service,item,raw=setup_table(tmp_path)
    base=tmp_path/"user-uploads"
    records=research_files.save_inputs(base,[("new.csv",raw),("notes.md",b"Public assay notes")])
    original=research_files.resolve_inputs(base,[record["id"] for record in records])
    staged=research_files.stage_inputs(base,work,[record["id"] for record in records],data_store=store)
    table=next(item for item in staged if item["name"]=="new.csv")
    assert store.original_bytes(table["path"])==raw
    assert original[0][0].read_bytes()==raw
    assert "9100000013" not in workspace_text(work)
    assert "Public assay notes" in workspace_text(work)
    # A later request gets its own store instance, as the API does.
    later=ResearchDataStore(store.root,work)
    repeated=research_files.stage_inputs(base,work,[records[0]["id"]],data_store=later)
    assert repeated[0]["path"]==table["path"]
    assert "9100000013" not in workspace_text(work)
    with pytest.raises(FileNotFoundError):
        research_files.stage_inputs(tmp_path/"another-owner",work,[records[0]["id"]],data_store=store)


@pytest.mark.asyncio
async def test_actual_sandbox_cannot_read_original_or_heldout_labels(tmp_path):
    from agent.harness.research_workspace import run_python
    work,store,service,item,raw=setup_table(tmp_path)
    private_file=store.root/"tables"/item["sha256"]
    probe=("from pathlib import Path\nimport json\n"
           "p=Path("+repr(str(private_file))+")\n"
           "try:\n p.read_bytes(); private_read=True\nexcept OSError:\n private_read=False\n"
           "print(json.dumps({'private_read':private_read,'table':Path("+repr(item["path"])+").read_text()}))\n")
    (work/"probe.py").write_text(probe)
    for prepared in [False,True]:
        if prepared:
            result=await prepare(service,item)
            data=json.loads(store.load_evaluation(result["evaluation_file"]))
        execution=await run_python(work,tmp_path/("after" if prepared else "before"),"probe.py",30)
        assert execution["exit_code"]==0,execution
        output=json.loads(execution["stdout"])
        assert output["private_read"] is False
        if prepared:
            assert all(marker not in output["table"] for marker in hidden_markers(data))
        else:
            assert "9100000013" not in output["table"]


@pytest.mark.asyncio
async def test_protected_fit_revision_review_delivery_and_private_reuse(tmp_path):
    from agent.harness.research_workspace import run_python,publish_proposals,run_capability
    from agent.harness.capability_review import review_proposal
    from agent.harness.research_development import build_delivery
    from test_research_development import review_spec
    import zipfile
    work,store,service,item,raw=setup_table(tmp_path)
    service.plan["deliverable"]="model"
    prepared=await prepare(service,item)
    data=json.loads(store.load_evaluation(prepared["evaluation_file"]))
    markers=hidden_markers(data)
    algorithm = """def fit(training, seed):
    assert set(training)=={'task','strategy','train','unlabeled'}
    return {'center':sum(r['target'] for r in training['train'])/len(training['train']),
            'offset':0.0}
def predict(model, inputs):
    if set(inputs)!={'sequence'} or not isinstance(inputs['sequence'],str) or len(inputs['sequence'])<2:
        raise ValueError('Provide a sequence')
    return {'value':model['center']+model['offset']}
"""
    await service.write_research_file("learner.py",algorithm)
    base=await service.train_research_model("mean","learner.py",prepared["evaluation_file"],
                                            "Synthetic training mean")
    assert base["training"]["validation_and_test_mounted"] is False
    await service.write_research_file("learner.py",algorithm.replace("'offset':0.0","'offset':0.125"))
    revision=await service.train_research_model("offset","learner.py","",
        "Synthetic revision to verify lifecycle isolation",parent_experiment_id=base["experiment"]["id"])
    e,t=revision["experiment"],revision["training"]
    assert e["parent"]==base["experiment"]["id"]
    assert e["fresh_process_replay"] and not e["test_used_for_feedback"]
    assert e["evaluation_visibility"]=="host_only"
    restored=await service.restore_research_evaluation(e["id"])
    assert restored["data_access"]=="host_only"
    assert set(json.loads((work/restored["path"]).read_text()))=={"schema","id"}
    assert all(marker not in workspace_text(work) for marker in markers)
    specification=review_spec(e["id"])
    specification.update(evaluation_file=t["evaluation_file"],
        reproduction_script=t["reproduction_script"],fitted_files=t["fitted_files"],
        evidence_files=t["evidence_files"]+["README.md"],dependencies=list(t["dependency_versions"]))
    specification["input_schema"]={"type":"object","required":["sequence"],
        "properties":{"sequence":{"type":"string","minLength":2}},"additionalProperties":False}
    expected=sum(row["target"] for row in data["splits"]["train"])/len(data["splits"]["train"])+0.125
    specification["cases"]=[{"inputs":{"sequence":"ACDEFG"},"expected":{"value":expected}},
                             {"inputs":{},"reject":True}]
    await service.write_research_file("README.md","Synthetic isolation fixture; no biological result.")
    await service.propose_research_capability("Isolation fixture","Synthetic fitted model",
        t["source"],t["fitted_files"],[{"sequence":"ACDEFG"}],specification)
    bundle=service.proposals[-1]
    assert json.loads((bundle/t["evaluation_file"]).read_text())["splits"]["test"]
    review_contexts=[]
    async def reviewer(context):
        review_contexts.append(context)
        assert all(marker not in json.dumps(context) for marker in markers)
        return {"decision":"approve","summary":"Software isolation fixture only.",
            "checks":{k:{"passed":True,"reason":"Controlled synthetic fixture"} for k in context["required_checks"]},
            "issues":[],"tests":[{"inputs":{"sequence":"GGAACCTT"},"invariants":{"finite_prediction":True}},
                                 {"inputs":{"sequence":""},"reject":True}]}
    review_runs=[]
    async def guarded_runner(workspace,control,script,timeout=120,**kwargs):
        review_runs.append(str(workspace))
        assert all(marker not in workspace_text(Path(workspace)) for marker in markers)
        return await run_python(workspace,control,script,timeout,**kwargs)

    receipt=await review_proposal(bundle,reviewer=reviewer,control=tmp_path/"admission",runner=guarded_runner)
    assert receipt["status"]=="approved",receipt
    assert review_contexts and review_runs
    assert receipt["automatic"]["evaluation"]["training_reproduced"]
    delivery=build_delivery(work,bundle,receipt,goal_id="isolation-fixture")
    with zipfile.ZipFile(work/delivery["package"]) as archive:
        for name in archive.namelist():
            content=archive.read(name).decode("utf-8",errors="ignore")
            assert all(marker not in content for marker in markers),name
        assert not any(name.endswith("evaluation.json") for name in archive.namelist())
    assert all(marker not in workspace_text(work) for marker in markers)
    library=tmp_path/"private_capabilities"
    entry=publish_proposals([bundle],"isolation-fixture",library)[0]
    result=await run_capability(entry["id"],{"sequence":"ACDEFG"},library=library)
    assert result["value"]==pytest.approx(expected)


@pytest.mark.parametrize("suffix",[".json",".jsonl",".xlsx",".parquet"])
@pytest.mark.asyncio
async def test_unsupported_table_formats_cannot_leak_values_in_parse_errors(tmp_path,suffix):
    work,store,service,item,original=setup_table(tmp_path,suffix)
    assert "9100000013" not in workspace_text(work)
    with pytest.raises(ValueError,match="accepts CSV/TSV") as error:
        await prepare(service,item)
    assert "9100000013" not in str(error.value)
    assert store.original_bytes(item["path"])==original


def test_large_local_table_is_streamed_and_protected_before_preparation(tmp_path,monkeypatch):
    from backend.web_v2 import research_files
    monkeypatch.setenv("WEBUI_V2_MODE","local")
    base=tmp_path/"uploads-owner";identity="a"*32
    folder=base/"uploads"/identity;folder.mkdir(parents=True)
    source=folder/"large.csv"
    with source.open("wb") as stream:
        stream.write(b"id,group,sequence,target\n")
        # Sparse data is a size/boundary fixture, not a valid biological dataset.
        stream.truncate(101*1024**2)
    digest=hashlib.sha256()
    with source.open("rb") as stream:
        while block:=stream.read(65536):digest.update(block)
    record={"id":identity,"name":source.name,"sha256":digest.hexdigest(),"bytes":source.stat().st_size}
    (folder/".metadata.json").write_text(json.dumps(record))
    work=tmp_path/"workspace";work.mkdir()
    store=ResearchDataStore(tmp_path/"private",work)
    # Resolution and protected staging must not materialize the whole upload.
    monkeypatch.setattr(research_files,"read_bytes",lambda *a,**k: (_ for _ in ()).throw(AssertionError("Whole-upload read")))
    items=research_files.stage_inputs(base,work,[identity],data_store=store)
    assert len(items)==1 and source.stat().st_size==101*1024**2
    marker=json.loads((work/items[0]["path"]).read_text())
    assert marker["columns"]==["id","group","sequence","target"]
    assert marker["sha256"]==record["sha256"]
    assert (store.root/"tables"/record["sha256"]).stat().st_size==source.stat().st_size
    store.protect_inputs(items,prior_agent_execution=True)
    assert (work/items[0]["path"]).stat().st_size<2048
    assert source.exists()


def test_changed_stream_cannot_commit_a_private_table_or_expose_rows(tmp_path):
    work=tmp_path/"workspace";work.mkdir()
    source=tmp_path/"source";source.mkdir()
    (source/"data.csv").write_bytes(b"id,target\nA,123456789\n")
    store=ResearchDataStore(tmp_path/"private",work)
    with pytest.raises(ValueError,match="upload receipt"):
        store.ingest_file("inputs/data.csv",source,"data.csv","f"*64)
    assert not (store.root/"tables"/("f"*64)).exists()
    assert not (work/"inputs/data.csv").exists()
    assert not store.entries
