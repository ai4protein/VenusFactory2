"""Real isolated fitting on synthetic advanced-learning protocol fixtures."""
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

from agent.harness.research_data_store import ResearchDataStore
from agent.harness.research_data_memory import prepared_dataset_context
from agent.harness.research_evaluation import validate_data
from agent.harness.research_workspace import ResearchWorkspace, run_python


def upload(store, path, fields, rows):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode()
    store.ingest(path, raw, hashlib.sha256(raw).hexdigest())
    return raw


def setup(tmp_path, *, classification=False, families=False, long_sequences=False):
    work = tmp_path/"workspace"
    store = ResearchDataStore(tmp_path/"private", work)
    episodes = tmp_path/"episodes"
    service = ResearchWorkspace(work, episodes/"cycle_000"/"tools", {"steps": [], "deliverable": "model"},
                                development_root=tmp_path/"development", data_store=store)
    rows = []
    for task in range(12):
        for i in range(8):
            rows.append({"id": f"T{task}-{i}", "group": f"F{task//2}" if families else f"T{task}",
                         "task": f"T{task}", "protein": "A"*(task+3)+("G"*700 if long_sequences else ""),
                         "x": (-1 if i%2 else 1)*(3+i/10+task/100) if classification else i,
                         "measurement": str(i%2) if classification else 999000000+task*101+2.125*i})
    raw = upload(store, "inputs/measured.csv", ["id", "group", "task", "protein", "x", "measurement"], rows)
    return work, store, service, episodes, rows, raw


def args(**kwargs):
    return dict(data_file="inputs/measured.csv", target_column="measurement", input_columns=["protein", "x"],
                group_column="group", id_column="id", provenance="Synthetic software protocol fixture",
                grouping="Whole independent tasks or connected family groups", seed=7, **kwargs)


def pool(store, *, bad=None):
    rows = [{"id": f"U{i}", "protein": "GGG", "x": (-1 if i%2 else 1)*(1+i/100),
             "private_note": "UNSELECTED_POOL_CONTENT"} for i in range(24)]
    fields = ["id", "protein", "x", "private_note"]
    if bad == "target":
        fields.append("measurement")
        for row in rows:
            row["measurement"] = 123
    if bad == "overlap_id":
        rows[0]["id"] = "T0-0"
    if bad == "overlap_input":
        rows[0].update(protein="AAA", x="0")
    if bad == "duplicate":
        rows[1] = dict(rows[0])
    return upload(store, "inputs/unlabeled.csv", fields, rows)


def all_text(folder):
    return "\n".join(path.read_bytes().decode("utf-8", errors="ignore") for path in folder.rglob("*") if path.is_file())


@pytest.mark.asyncio
@pytest.mark.parametrize("families", [False, True])
async def test_episodes_preserve_all_rows_and_keep_tasks_and_families_disjoint(tmp_path, families):
    work, store, service, episodes, source, raw = setup(tmp_path, families=families)
    result = await service.prepare_research_dataset(**args(task_column="task", support_size=2))
    data = json.loads(store.load_evaluation(result["evaluation_file"]))
    assert validate_data(data, "meta_learning") == data
    assert sum(len(data[part][split]) for part in ["splits", "support"] for split in data[part]) == len(source)
    ids = [row["id"] for part in ["splits", "support"] for rows in data[part].values() for row in rows]
    assert set(ids) == {row["id"] for row in source} and len(ids) == len(set(ids))
    tasks = [{row["task_id"] for row in rows} for rows in data["splits"].values()]
    assert all(not a & b for i, a in enumerate(tasks) for b in tasks[i+1:])
    visible = list(csv.DictReader(io.StringIO((work/"inputs/measured.csv").read_text())))
    train_tasks = {row["task_id"] for row in data["splits"]["train"]}
    assert {row["task"] for row in visible} == train_tasks
    assert len(visible) == result["training_data"]["rows"]
    assert result["training_data"]["scope"] == "training_tasks_only"
    hidden = [str(row["target"]) for split in ["validation", "test"]
              for part in ["splits", "support"] for row in data[part][split]]
    assert all(value not in all_text(work) for value in hidden)
    assert all(value not in json.dumps(result) for value in hidden)
    assert result["model_interface"]["shape_example"]["prediction_inputs"]["support"]
    assert result["episodes"]["support_size"] == 2
    restored = prepared_dataset_context(work, episodes, data_store=store)
    assert restored["episodes"] == result["episodes"]
    assert restored["model_interface"] == result["model_interface"]
    assert store.original_bytes("inputs/measured.csv") == raw
    again = await service.prepare_research_dataset(**args(task_column="task", support_size=2))
    assert again["evaluation_sha256"] == result["evaluation_sha256"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [{"support_size": 3}, {"seed": 9}, {"task_column": "", "support_size": 0}])
async def test_episode_role_or_split_changes_cannot_reveal_a_new_partition(tmp_path, change):
    work, store, service, _, _, _ = setup(tmp_path)
    kw = args(task_column="task", support_size=2)
    await service.prepare_research_dataset(**kw)
    before = (work/"inputs/measured.csv").read_bytes()
    with pytest.raises(ValueError, match="original held-out partition"):
        await service.prepare_research_dataset(**{**kw, **change, "output_file": "changed.json"})
    assert not (work/"changed.json").exists()
    assert (work/"inputs/measured.csv").read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["own_target", "support_id", "support_target", "support_input", "missing_support", "missing_query"])
async def test_episode_bindings_cannot_be_changed_before_scoring(tmp_path, damage):
    _, store, service, _, _, _ = setup(tmp_path)
    result = await service.prepare_research_dataset(**args(task_column="task", support_size=2))
    data = json.loads(store.load_evaluation(result["evaluation_file"]))
    row = data["splits"]["test"][0]
    if damage == "own_target":
        row["inputs"]["support"][0]["target"] = row["target"]
    elif damage == "support_id":
        data["support"]["test"][0]["id"] = data["splits"]["train"][0]["id"]
    elif damage == "support_target":
        data["support"]["test"][0]["target"] = "not a measured number"
    elif damage == "support_input":
        data["support"]["test"][0]["inputs"] = {k: v for k, v in row["inputs"].items() if k != "support"}
    elif damage == "missing_support":
        del data["support"]
    else:
        data["splits"]["test"] = data["splits"]["test"][:2]
    with pytest.raises(ValueError):
        validate_data(data, "meta_learning")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", ["target", "overlap_id", "overlap_input", "duplicate"])
async def test_invalid_unlabeled_pool_exposes_neither_table(tmp_path, bad):
    work, _, service, _, _, _ = setup(tmp_path)
    store = service.data_store
    pool(store, bad=bad)
    before = {name: (work/name).read_bytes() for name in ["inputs/measured.csv", "inputs/unlabeled.csv"]}
    with pytest.raises(ValueError):
        await service.prepare_research_dataset(**args(unlabeled_file="inputs/unlabeled.csv", unlabeled_id_column="id"))
    assert all((work/name).read_bytes() == content for name, content in before.items())
    assert not (work/"data").exists()


@pytest.mark.asyncio
async def test_unlabeled_pool_is_projected_and_verified_on_resume(tmp_path):
    work, store, service, episodes, _, _ = setup(tmp_path)
    original = pool(store)
    result = await service.prepare_research_dataset(**args(task_column="task", support_size=2,
        unlabeled_file="inputs/unlabeled.csv", unlabeled_id_column="id"))
    data = json.loads(store.load_evaluation(result["evaluation_file"]))
    assert len(data["unlabeled"]) == 24
    assert all(set(row) == {"id", "inputs"} for row in data["unlabeled"])
    assert "UNSELECTED_POOL_CONTENT" not in (work/"inputs/unlabeled.csv").read_text()
    assert set(next(csv.reader(io.StringIO((work/"inputs/unlabeled.csv").read_text())))) == {"id", "protein", "x"}
    assert store.original_bytes("inputs/unlabeled.csv") == original
    restored = prepared_dataset_context(work, episodes, data_store=store)
    assert restored["unlabeled_data"] == result["unlabeled_data"]
    assert restored["unlabeled_pool"] == result["unlabeled_pool"]
    private = store.root/"tables"/hashlib.sha256(original).hexdigest()
    private.write_bytes(b"changed")
    assert prepared_dataset_context(work, episodes, data_store=store) is None


@pytest.mark.asyncio
async def test_second_input_partition_conflict_does_not_expose_first_input(tmp_path):
    work, store, service, _, _, _ = setup(tmp_path)
    pool(store)
    store.entries["inputs/unlabeled.csv"]["partition"] = "previously-selected-other-view"
    store._save()
    before = (work/"inputs/measured.csv").read_bytes()
    with pytest.raises(ValueError, match="original held-out partition"):
        await service.prepare_research_dataset(**args(unlabeled_file="inputs/unlabeled.csv", unlabeled_id_column="id"))
    assert (work/"inputs/measured.csv").read_bytes() == before


BASELINE = """def fit(training, seed):
    return {'mean':sum(row['target'] for row in training['train'])/len(training['train'])}
def predict(model, inputs):
    return {'value':model['mean']}
"""
META = """def fit(training, seed):
    tasks = {}
    for row in training['train']:
        tasks.setdefault(row['task_id'], []).append(row)
    slopes = []
    for rows in tasks.values():
        x = [float(row['inputs']['x']) for row in rows]
        y = [row['target'] for row in rows]
        mx, my = sum(x)/len(x), sum(y)/len(y)
        slopes.append(sum((a-mx)*(b-my) for a,b in zip(x,y))/sum((a-mx)**2 for a in x))
    return {'slope_prior':sum(slopes)/len(slopes)}
def predict(model, inputs):
    support = inputs['support']
    if len(support)<2:
        raise ValueError('Provide two independent support observations')
    slope = model['slope_prior']
    offset = sum(row['target']-slope*float(row['inputs']['x']) for row in support)/len(support)
    return {'value':offset+slope*float(inputs['x'])}
"""


@pytest.mark.asyncio
async def test_meta_fit_revision_independent_review_zip_and_private_agent_call(tmp_path):
    from agent.harness.capability_review import review_proposal
    from agent.harness.research_development import build_delivery
    from agent.harness.research_workspace import publish_proposals, run_capability
    from test_research_development import review_spec

    work, store, service, _, _, _ = setup(tmp_path, long_sequences=True)
    prepared = await service.prepare_research_dataset(**args(task_column="task", support_size=2))
    data = json.loads(store.load_evaluation(prepared["evaluation_file"]))
    hidden_queries = [str(row["target"]) for split in ["validation", "test"] for row in data["splits"][split]]
    await service.write_research_file("baseline.py", BASELINE)
    base = await service.train_research_model("mean", "baseline.py", prepared["evaluation_file"], "No task adaptation")
    await service.write_research_file("meta.py", META)
    child = await service.train_research_model("shared slope prior", "meta.py", "", "Fit a shared prior, adapt offsets using same-task support",
        strategy="meta_learning", parent_experiment_id=base["experiment"]["id"])
    t, e = child["training"], child["experiment"]
    assert e["comparison"]["improved_on_validation"] and e["validation"]["score"] < 1e-6
    assert e["validation"]["episodes"]["score"]["macro_score"] < 1e-6
    assert e["validation"]["episodes"]["references"]["support_median"]["macro_score"] > 0
    assert e["validation"]["episodes"]["task_count"] == 2
    training = json.loads((work/t["reproduction_script"]).with_name("training.json").read_text())
    assert {row["task_id"] for row in training["train"]} == {row["task_id"] for row in data["splits"]["train"]}
    assert all(value not in json.dumps(training) for value in hidden_queries)
    example = {"protein": "GGG", "x": "2", "support": [
        {"id": "a", "inputs": {"protein": "GGG", "x": "0"}, "target": 1.0},
        {"id": "b", "inputs": {"protein": "GGG", "x": "1"}, "target": 3.125}]}
    spec = review_spec(e["id"])
    spec.update(evaluation_file=t["evaluation_file"], reproduction_script=t["reproduction_script"],
        fitted_files=t["fitted_files"], evidence_files=t["evidence_files"]+["README.md"], dependencies=list(t["dependency_versions"]))
    spec["input_schema"] = {"type": "object", "required": ["protein", "x", "support"],
        "properties": {"protein": {"type": "string"}, "x": {"type": "string"},
                       "support": {"type": "array", "minItems": 2}}}
    spec["cases"] = [{"inputs": example, "expected": {"value": 5.25}}, {"inputs": {}, "reject": True}]
    await service.write_research_file("README.md", "Synthetic task adaptation fixture; software validation only.")
    await service.propose_research_capability("Task adaptation fixture", "Synthetic learned prior and support adaptation",
        t["source"], t["fitted_files"], [example], spec)
    bundle = service.proposals[-1]
    async def reviewer(context):
        assert all(value not in json.dumps(context) for value in hidden_queries)
        training_docs = [item for item in context["documents"] if item.get("kind") == "training_data"]
        assert len(training_docs) == 1 and "text" not in training_docs[0]
        assert training_docs[0]["training_rows"] == len(data["splits"]["train"])
        assert len(json.dumps(training_docs)) < 2000
        episodes = context["host_evidence"]["evaluation"]["episodes"]
        assert episodes["task_count"] == 2 and episodes["score"]["macro_score"] < 1e-6
        assert episodes["references"]["support_mean"]["macro_score"] > 0
        return {"decision": "approve", "summary": "Synthetic protocol test only", "issues": [],
            "checks": {key: {"passed": True, "reason": "Controlled software fixture"} for key in context["required_checks"]},
            "tests": [{"inputs": {**example, "x": "3"}, "invariants": {"finite_prediction": True}},
                      {"inputs": {"protein": "GGG", "x": "2"}, "reject": True}]}
    async def guarded_runner(workspace, control, script, timeout=120, **kwargs):
        assert all(value not in all_text(Path(workspace)) for value in hidden_queries)
        return await run_python(workspace, control, script, timeout, **kwargs)
    receipt = await review_proposal(bundle, reviewer=reviewer, control=tmp_path/"review", runner=guarded_runner)
    assert receipt["status"] == "approved", receipt
    assert receipt["automatic"]["evaluation"]["training_reproduced"]
    delivery = build_delivery(work, bundle, receipt, goal_id="meta-protocol")
    with zipfile.ZipFile(work/delivery["package"]) as archive:
        assert all(value not in archive.read(name).decode("utf-8", errors="ignore")
                   for value in hidden_queries for name in archive.namelist())
    with zipfile.ZipFile(work/delivery["model_package"]) as archive:
        text = archive.read("evaluation/test-report.md").decode()
        assert "Few-shot comparisons" in text and "Equal-task score" in text
        assert "Support mean" in text and "Support median" in text
        report = json.loads(archive.read("evaluation/test-report.json"))
        assert report["final_test"]["episodes"]["score"]["macro_score"] < 1e-6
        assert report["validation_trajectory"]["generations"][1]["validation"]["episodes"]["task_count"] == 2
    library = tmp_path/"private-capabilities"
    entry = publish_proposals([bundle], "meta-protocol", library)[0]
    prediction = await run_capability(entry["id"], example, library=library)
    assert prediction["value"] == pytest.approx(5.25)


@pytest.mark.asyncio
async def test_real_semisupervised_fit_and_revision_use_the_same_queries(tmp_path):
    work, store, service, _, _, _ = setup(tmp_path, classification=True)
    pool(store)
    prepared = await service.prepare_research_dataset(**args(task="classification",
        unlabeled_file="inputs/unlabeled.csv", unlabeled_id_column="id"))
    await service.write_research_file("baseline.py", """from collections import Counter
def fit(training, seed):
    return {'majority': Counter(row['target'] for row in training['train']).most_common(1)[0][0]}
def predict(model, inputs):
    return {'value': model['majority']}
""")
    base = await service.train_research_model("majority", "baseline.py", prepared["evaluation_file"], "Labeled-only baseline")
    await service.write_research_file("semi.py", """from sklearn.semi_supervised import LabelSpreading
def fit(training, seed):
    labeled, unlabeled = training['train'], training['unlabeled']
    assert unlabeled and all(set(row)=={'id','inputs'} for row in unlabeled)
    x = [[float(row['inputs']['x'])] for row in labeled+unlabeled]
    y = [int(row['target']) for row in labeled]+[-1]*len(unlabeled)
    return LabelSpreading(kernel='rbf', gamma=5).fit(x,y)
def predict(model, inputs):
    return {'value':str(int(model.predict([[float(inputs['x'])]])[0]))}
""")
    child = await service.train_research_model("label spreading", "semi.py", "", "Use a compatible unlabeled neighborhood",
        strategy="semi_supervised", parent_experiment_id=base["experiment"]["id"])
    assert child["experiment"]["evaluation_sha256"] == base["experiment"]["evaluation_sha256"]
    assert child["experiment"]["validation"]["score"] == 1.0
    assert child["experiment"]["comparison"]["improved_on_validation"]
    assert child["training"]["validation_and_test_mounted"] is False


@pytest.mark.parametrize("damage", ["target", "heldout", "strategy", "extra_field"])
def test_training_evidence_must_match_the_frozen_training_view(tmp_path, damage):
    from agent.harness.capability_review import training_document
    from agent.harness.research_evaluation import training_view
    from test_research_data_preflight import dataset

    data = dataset()
    value = copy.deepcopy(training_view(data, "supervised"))
    if damage == "target":
        value["train"][0]["target"] += 1
    elif damage == "heldout":
        value["train"].extend(data["splits"]["test"])
    elif damage == "strategy":
        value["strategy"] = "meta_learning"
    else:
        value["test"] = data["splits"]["test"]
    (tmp_path/"evaluation.json").write_text(json.dumps(data))
    (tmp_path/"training.json").write_text(json.dumps(value))
    manifest = {"review_spec": {"kind": "model", "evaluation_file": "evaluation.json"}}
    with pytest.raises(ValueError, match="training-only data"):
        training_document(tmp_path, manifest, tmp_path/"training.json")


@pytest.mark.asyncio
async def test_consistent_support_edit_still_cannot_overlap_a_heldout_query(tmp_path):
    _, store, service, _, _, _ = setup(tmp_path)
    prepared = await service.prepare_research_dataset(**args(task_column="task", support_size=2))
    data = json.loads(store.load_evaluation(prepared["evaluation_file"]))
    support = data["support"]["test"][0]
    query = next(row for row in data["splits"]["test"] if row["task_id"] == support["task_id"])
    support["inputs"] = {key: value for key, value in query["inputs"].items() if key != "support"}
    for row in data["splits"]["test"]:
        for observation in row["inputs"]["support"]:
            if observation["id"] == support["id"]:
                observation["inputs"] = dict(support["inputs"])
    with pytest.raises(ValueError, match="overlap within a task"):
        validate_data(data, "meta_learning")


@pytest.mark.asyncio
@pytest.mark.parametrize("extra", [{"task_column": "task"}, {"support_size": 2},
    {"task_column": "task", "support_size": 8}, {"unlabeled_id_column": "id"}])
async def test_invalid_advanced_selectors_do_not_expose_measurements(tmp_path, extra):
    work, _, service, _, _, _ = setup(tmp_path)
    before = (work/"inputs/measured.csv").read_bytes()
    with pytest.raises(ValueError):
        await service.prepare_research_dataset(**args(**extra))
    assert (work/"inputs/measured.csv").read_bytes() == before
    assert not (work/"data").exists()


@pytest.mark.asyncio
async def test_support_expansion_size_limit_applies_before_publishing_views(tmp_path, monkeypatch):
    work = tmp_path/"workspace"
    store = ResearchDataStore(tmp_path/"private", work)
    service = ResearchWorkspace(work, tmp_path/"control", {"steps": []}, data_store=store)
    rows = [{"id": f"T{task}-{i}", "group": f"T{task}", "task": f"T{task}",
             "protein": "A"*(task+3)+"G"*90000, "x": i, "measurement": task*10+i}
            for task in range(12) for i in range(8)]
    upload(store, "inputs/measured.csv", ["id", "group", "task", "protein", "x", "measurement"], rows)
    before = (work/"inputs/measured.csv").read_bytes()
    original_dumps = json.dumps
    def bounded_only(value, *positional, **keywords):
        if isinstance(value, dict) and "support" in value and "splits" in value:
            raise AssertionError("Do not allocate the entire expanded evaluation before checking its size")
        return original_dumps(value, *positional, **keywords)
    monkeypatch.setattr(json, "dumps", bounded_only)
    with pytest.raises(ValueError, match="Prepared evaluation exceeds 20 MiB"):
        await service.prepare_research_dataset(**args(task_column="task", support_size=2))
    assert (work/"inputs/measured.csv").read_bytes() == before
    assert not (work/"data").exists()


def test_progressive_tool_discovery_exposes_advanced_arguments_on_request(tmp_path):
    from agent.harness.tool_disclosure import ToolDisclosure

    _, _, service, _, _, _ = setup(tmp_path)
    tools = {tool.name: tool for tool in service.tools()}
    index = ToolDisclosure(tools, schema=lambda tool: tool.get_input_schema().model_json_schema(),
                           allowed=lambda tool: tool.name in tools)
    found = index.metadata("search_tools", {"query": "prepare_research_dataset"})
    card = next(tool for tool in found["tools"] if tool["name"] == "prepare_research_dataset")
    assert set(card) == {"name", "description"}
    schema = tools["prepare_research_dataset"].get_input_schema().model_json_schema()
    assert {"task_column", "support_size", "unlabeled_file", "unlabeled_id_column"} <= set(schema["properties"])
    described = index.metadata("describe_tool", {"name": "prepare_research_dataset"})
    assert all(name in json.dumps(described) for name in ["task_column", "support_size", "unlabeled_file"])
