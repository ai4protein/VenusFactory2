"""Author-provided study partitions stay fixed and hidden behind the host store."""
import csv
import hashlib
import io
import json

import pytest

from agent.harness.research_data_memory import prepared_dataset_context
from agent.harness.research_data_store import ResearchDataStore
from agent.harness.research_dataset import prepare_dataset
from agent.harness.research_workspace import ResearchWorkspace


def upload(store, rows, path="inputs/measured.csv"):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode()
    store.ingest(path, raw, hashlib.sha256(raw).hexdigest())
    return raw


def fixture(tmp_path, *, damage=None):
    work = tmp_path / "workspace"
    store = ResearchDataStore(tmp_path / "private", work)
    episodes = tmp_path / "episodes"
    service = ResearchWorkspace(work, episodes / "cycle_000" / "tools", {"steps": [], "deliverable": "model"},
                                development_root=tmp_path / "development", data_store=store)
    rows = [{"id": f"T{task}-{n}", "group": f"F{task // 2}", "task": f"T{task}",
             "protein": "A" * (task + 3), "x": str(n), "measurement": str(999000000 + task * 101 + n * 2.125),
             "fold": "train" if task < 8 else "validation" if task < 10 else "test"}
            for task in range(12) for n in range(8)]
    if damage:
        damage(rows)
    raw = upload(store, rows)
    return work, store, service, episodes, rows, raw


def args(**extra):
    return dict(data_file="inputs/measured.csv", target_column="measurement", input_columns=["protein", "x"],
                group_column="group", id_column="id", provenance="Synthetic protocol fixture",
                grouping="Paired tasks share a biological family", seed=7, split_column="fold", **extra)


@pytest.mark.asyncio
@pytest.mark.parametrize("episodic", [False, True])
async def test_author_splits_preserve_measurements_without_exposing_holdouts(tmp_path, episodic):
    work, store, service, episodes, rows, raw = fixture(tmp_path)
    kw = args(**({"task_column": "task", "support_size": 2} if episodic else {}))
    result = await service.prepare_research_dataset(**kw)
    data = json.loads(store.load_evaluation(result["evaluation_file"]))
    for name in ["train", "validation", "test"]:
        actual = data["splits"][name] + data.get("support", {}).get(name, [])
        assert {row["id"] for row in actual} == {row["id"] for row in rows if row["fold"] == name}
        assert all("fold" not in row["inputs"] for row in actual)
    visible = list(csv.DictReader(io.StringIO((work / "inputs/measured.csv").read_text())))
    assert len(visible) == 64 and {row["fold"] for row in visible} == {"train"}
    hidden = [row["measurement"] for row in rows if row["fold"] != "train"]
    workspace_text = "\n".join(p.read_text(errors="ignore") for p in work.rglob("*") if p.is_file())
    assert all(value not in workspace_text and value not in json.dumps(result) for value in hidden)
    assert result["partitioning"] == {"method": "provided_column", "column": "fold"}
    restored = prepared_dataset_context(work, episodes, data_store=store)
    assert restored["partitioning"] == result["partitioning"]
    assert store.original_bytes("inputs/measured.csv") == raw
    repeated = await service.prepare_research_dataset(**kw)
    assert repeated["evaluation_sha256"] == result["evaluation_sha256"]


@pytest.mark.asyncio
@pytest.mark.parametrize("damage,match", [
    (lambda rows: rows[0].update(fold=""), "train, validation or test"),
    (lambda rows: rows[0].update(fold="holdout"), "train, validation or test"),
    (lambda rows: rows[0].update(fold="test"), "biological group"),
    (lambda rows: [row.update(fold="train") for row in rows if row["fold"] == "validation"], "all three partitions"),
])
async def test_invalid_assignments_never_publish_training_data(tmp_path, damage, match):
    work, _, service, _, _, _ = fixture(tmp_path, damage=damage)
    before = (work / "inputs/measured.csv").read_bytes()
    with pytest.raises(ValueError, match=match):
        await service.prepare_research_dataset(**args())
    assert (work / "inputs/measured.csv").read_bytes() == before
    assert not (work / "data").exists()


@pytest.mark.asyncio
async def test_a_task_connecting_families_cannot_cross_provided_splits(tmp_path):
    def damage(rows):
        for row in rows:
            if row["task"] == "T8":
                row["task"] = "T0"
    work, _, service, _, _, _ = fixture(tmp_path, damage=damage)
    before = (work / "inputs/measured.csv").read_bytes()
    with pytest.raises(ValueError, match="connected task/group"):
        await service.prepare_research_dataset(**args(task_column="task", support_size=2))
    assert (work / "inputs/measured.csv").read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("change,match", [
    ({"input_columns": ["protein", "fold"]}, "partition column"),
    ({"split_column": "measurement"}, "partition column"),
    ({"split_column": "group"}, "partition column"),
    ({"split_column": "unknown"}, "Selected columns are missing"),
    ({"split_column": None}, "column names"),
])
async def test_partition_selector_cannot_be_a_feature_or_another_role(tmp_path, change, match):
    _, _, service, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match=match):
        await service.prepare_research_dataset(**{**args(), **change})


@pytest.mark.asyncio
async def test_repreparing_original_table_cannot_reassign_hidden_rows(tmp_path):
    work, _, service, _, _, _ = fixture(tmp_path)
    await service.prepare_research_dataset(**args())
    before = (work / "inputs/measured.csv").read_bytes()
    with pytest.raises(ValueError, match="original held-out partition"):
        await service.prepare_research_dataset(**{**args(), "split_column": "", "seed": 123})
    assert (work / "inputs/measured.csv").read_bytes() == before


def test_training_curation_preserves_existing_episode_queries_and_support(tmp_path):
    _, _, _, _, rows, _ = fixture(tmp_path)
    original = tmp_path / "original"
    original.mkdir()
    (original / "inputs").mkdir()
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (original / "inputs/measured.csv").write_text(stream.getvalue())
    kw = {**args(task_column="task", support_size=2), "split_column": ""}
    before = prepare_dataset(original, **kw)
    data = json.loads((original / before["evaluation_file"]).read_text())
    membership = {row["id"]: name for part in ["splits", "support"] for name, items in data[part].items() for row in items}
    excluded = data["splits"]["train"][0]["task_id"]
    filtered = [{**row, "fold": membership[row["id"]]} for row in rows if row["task"] != excluded]
    work = tmp_path / "curated"
    work.mkdir()
    store = ResearchDataStore(tmp_path / "curated_private", work)
    upload(store, filtered)
    after = prepare_dataset(work, **args(task_column="task", support_size=2), data_store=store)
    curated = json.loads(store.load_evaluation(after["evaluation_file"]))
    for name in ["validation", "test"]:
        assert curated["splits"][name] == data["splits"][name]
        assert curated["support"][name] == data["support"][name]
    assert excluded not in {row["task_id"] for row in curated["splits"]["train"]}


def test_full_mcp_schema_discloses_optional_split_column(tmp_path):
    from agent.harness.tool_disclosure import ToolDisclosure
    _, _, service, _, _, _ = fixture(tmp_path)
    tools = {tool.name: tool for tool in service.tools()}
    index = ToolDisclosure(tools, schema=lambda tool: tool.get_input_schema().model_json_schema(), allowed=lambda tool: tool.name in tools)
    found = index.metadata("search_tools", {"query": "prepare_research_dataset"})
    card = next(tool for tool in found["tools"] if tool["name"] == "prepare_research_dataset")
    assert set(card) == {"name", "description"}
    described = index.metadata("describe_tool", {"name": "prepare_research_dataset"})
    assert "split_column" in json.dumps(described)


@pytest.mark.asyncio
async def test_provided_membership_ignores_seeded_holdout_fractions(tmp_path):
    _, store, service, _, _, _ = fixture(tmp_path)
    result = await service.prepare_research_dataset(**{**args(), "validation_fraction": 0.8, "test_fraction": 0.8})
    data = json.loads(store.load_evaluation(result["evaluation_file"]))
    assert {name: len(rows) for name, rows in data["splits"].items()} == {"train": 64, "validation": 16, "test": 16}
    assert "validation_fraction" not in data["preparation"]
    assert "test_fraction" not in data["preparation"]
