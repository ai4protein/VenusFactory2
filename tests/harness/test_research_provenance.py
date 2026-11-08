"""Metadata remains readable; any measurement-bearing JSON stays host-owned."""
import hashlib
import json
import pytest
from agent.harness.research_data_store import ResearchDataStore, provenance_only


def provenance():
    return {"dataset":"public/fixture", "url":"https://example.org/dataset",
            "revision":"version-1", "source_file":"train.csv", "sha256":"a"*64,
            "rows":600, "fields":["sequence","target"], "license_from_dataset_card":"Apache-2.0",
            "original_validation_and_test_supplied":False, "acquisition":"Public metadata fixture"}


@pytest.mark.parametrize("streaming", [False, True])
def test_original_provenance_stays_readable_and_hash_bound_on_resume(tmp_path, streaming):
    work=tmp_path/"work";(work/"inputs").mkdir(parents=True)
    path="inputs/provenance.json";raw=json.dumps(provenance(),indent=2).encode()
    (work/path).write_bytes(raw)
    store=ResearchDataStore(tmp_path/"private",work);sha=hashlib.sha256(raw).hexdigest()
    if streaming:
        store.ingest_file(path,work,path,sha)
    else:
        store.ingest(path,raw,sha)
    assert (work/path).read_bytes()==raw and store.original_bytes(path)==raw
    assert store.entries[path]["partition"] is None
    (work/path).write_text('{"url":"fabricated"}')
    resumed=ResearchDataStore(store.root,work)
    resumed.protect_inputs([{"path":path,"sha256":sha}],prior_agent_execution=True)
    assert (work/path).read_bytes()==raw
    assert not list((store.root/"evaluations").glob("*.json"))


@pytest.mark.parametrize("patch", [
    {"records":[{"target":"HELDOUT_SENTINEL"}]},
    {"splits":{"test":[{"target":"HELDOUT_SENTINEL"}]}},
    {"target":"HELDOUT_SENTINEL"},
    {"rows":[{"target":"HELDOUT_SENTINEL"}]},
    {"fields":[{"target":"HELDOUT_SENTINEL"}]},
    {"dataset":{"target":"HELDOUT_SENTINEL"}},
    {"rows":True},
    {"url":""},
    {"description":"HELDOUT_SENTINEL"*5000},
])
def test_metadata_like_objects_with_measurements_or_invalid_types_stay_protected(tmp_path, patch):
    value={**provenance(),**patch};raw=json.dumps(value).encode()
    assert not provenance_only(raw)
    work=tmp_path/"work";work.mkdir()
    store=ResearchDataStore(tmp_path/"private",work)
    store.ingest("data.json",raw,hashlib.sha256(raw).hexdigest())
    view=(work/"data.json").read_text()
    assert json.loads(view)["data_access"]=="protected_research_table"
    assert "HELDOUT_SENTINEL" not in view
    assert store.original_bytes("data.json")==raw


@pytest.mark.parametrize("raw", [
    b'[{"dataset":"fixture","target":"HELDOUT_SENTINEL"}]',
    b'{"sequence":"AAA","value":"HELDOUT_SENTINEL"}',
    b'{"invalid":',
    b'\xff\xfe',
])
def test_unrecognized_json_is_not_released_as_metadata(tmp_path, raw):
    assert not provenance_only(raw)
    work=tmp_path/"work";work.mkdir()
    store=ResearchDataStore(tmp_path/"private",work)
    store.ingest("data.json",raw,hashlib.sha256(raw).hexdigest())
    assert json.loads((work/"data.json").read_text())["data_access"]=="protected_research_table"


def test_private_provenance_hash_is_verified_before_restoring(tmp_path):
    work=tmp_path/"work";work.mkdir()
    store=ResearchDataStore(tmp_path/"private",work)
    raw=json.dumps(provenance()).encode();sha=hashlib.sha256(raw).hexdigest()
    store.ingest("provenance.json",raw,sha)
    (store.root/"tables"/sha).write_text('{"dataset":"tampered","url":"https://example.org"}')
    with pytest.raises(ValueError,match="hash changed"):
        store.restore_view("provenance.json")
    assert (work/"provenance.json").read_bytes()==raw
