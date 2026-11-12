"""Archive/recovery integration using synthetic publication storage fixtures."""

import json

import pytest
from test_model_retention import POLICY, record
from test_scientist_model_library import storage as _publication_storage

from agent.harness.contracts import digest
from agent.rsi.engine import model_curation, model_retention
from agent.rsi.engine.scientist_model_library import model_key
from mcp_tools.models.evolved.catalog import recorded_metadata

storage = _publication_storage


@pytest.fixture
def published_pair(storage, monkeypatch):
    library, root, verified, _ = storage
    first = library.publish(root, "b" * 64)["entry_id"]
    verified["packet"]["model"]["recipe"]["source_code"] += "\nMODEL_VARIANT = 1\n"
    verified["key"] = model_key(verified["packet"]["model"], None)
    second = library.publish(root, "b" * 64)["entry_id"]

    def comparison(lib, entry, policy):
        assert lib.root == library.root
        return record(entry, error=1 if entry == first else 2, size=10 if entry == first else 100)["comparison"]

    monkeypatch.setattr(model_retention, "_comparison", comparison)
    return library, first, second


def test_archive_filters_search_but_preserves_pinned_publications(published_pair):
    library, first, second = published_pair
    full = library.catalog()
    before = library.inspect_publication(second)
    assert recorded_metadata(library)["indexed_entries"] == 2
    plan = model_retention.plan_retention(library, POLICY)
    assert [x["entry_id"] for x in plan["archive"]] == [second]
    state = model_curation.apply(library, plan)
    assert state["archived"][second]["replacement"] == first
    assert recorded_metadata(library)["indexed_entries"] == 1
    assert library.catalog() == full
    assert library.inspect_publication(second) == before
    assert model_curation.apply(library, plan) == state
    restored = model_curation.restore(library, second, reason="Reproduce a pinned result")
    assert restored["archived"] == {}
    assert recorded_metadata(library)["indexed_entries"] == 2
    assert len(list((library.root / "curation/events").glob("*.json"))) == 2


def test_modified_plan_is_rejected(published_pair):
    library, _, _ = published_pair
    plan = model_retention.plan_retention(library, POLICY)
    plan["archive"] = []
    with pytest.raises(ValueError, match="plan changed"):
        model_curation.apply(library, plan)
    assert not (library.root / "curation/current.json").exists()


def test_tampered_state_is_rejected(published_pair):
    library, _, second = published_pair
    model_curation.apply(library, model_retention.plan_retention(library, POLICY))
    path = library.root / "curation/current.json"
    state = json.loads(path.read_text())
    state["archived"].pop(second)
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="state changed"):
        recorded_metadata(library)


def test_removed_keeper_reveals_original_model(published_pair):
    library, first, second = published_pair
    model_curation.apply(library, model_retention.plan_retention(library, POLICY))
    entry = library._object("entries", first)
    (library.root / "index" / (digest(entry["key"]) + ".json")).unlink()
    assert second not in model_curation.effective_archived(library)
    assert [x["entry_id"] for x in library.discovery_metadata()["models"]] == [second]


def test_restore_rejects_path_traversal_and_empty_reason(published_pair):
    library, _, second = published_pair
    with pytest.raises(ValueError):
        model_curation.restore(library, "../anything", reason="fixture")
    with pytest.raises(ValueError):
        model_curation.restore(library, second, reason="")
