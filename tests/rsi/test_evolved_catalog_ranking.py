"""RSI catalog ranking uses synthetic metadata, not scientific model evidence."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from mcp_tools.models.evolved import service as evolved
from mcp_tools.tools.discovery import service as discovery


def card(number, protein, *, assay="ADH1 reference measurements", **extra):
    return {
        "entry_id": format(number, "064x"),
        "scope": {"protein": protein, "target": "expression response", "assay": assay},
        "interface": "generated_context.v3",
        "stage": "expression",
        "runtime_compatible": True,
        "verified_now": False,
        **extra,
    }


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    from mcp_tools.tools import capability_policy, runtime_tool_policy

    store = capability_policy.CapabilityStore(tmp_path / "capabilities.sqlite3")
    monkeypatch.setattr(capability_policy, "CapabilityStore", lambda *a, **k: store)
    monkeypatch.setattr(runtime_tool_policy, "is_tool_allowed", lambda name: True)
    monkeypatch.setattr(discovery, "is_local_mode", lambda: True)
    monkeypatch.setattr(discovery, "platform_models", lambda *a, **k: [])
    cards = [card(i, f"background-{i}") for i in range(1, 10001)]
    cards.append(card(20000, "ADH1"))
    library = SimpleNamespace()
    monkeypatch.setattr(evolved, "catalog_cards", lambda _: (
        deepcopy(cards), {"indexed_entries": len(cards), "counting_rule": "synthetic fixtures"}
    ))
    monkeypatch.setattr(discovery, "model_cards", lambda: deepcopy(cards))
    return cards, store, library


def page(library, *, query="ADH1 expression", offset=0, limit=2):
    return evolved.list_models(library, query=query, stage="expression", limit=limit, offset=offset)


def test_named_protein_is_not_buried_by_ten_thousand_description_matches(catalog):
    cards, _, library = catalog
    first = page(library)
    assert first["models"][0]["entry_id"] == cards[-1]["entry_id"]
    assert first["matching_entries"] == 10001 and first["next_offset"] == 2
    assert len(first["models"]) == 2
    assert all("scope" not in row and "inputSchema" not in row for row in first["models"])
    second = page(library, offset=2)
    assert len(second["models"]) == 2
    assert not {row["entry_id"] for row in first["models"]} & {
        row["entry_id"] for row in second["models"]
    }


def test_policy_runtime_and_stage_filters_precede_ranking_and_pagination(catalog):
    cards, store, library = catalog
    wanted = cards[-1]["entry_id"]
    cards[:0] = [
        card(20001, "ADH1", runtime_compatible=False),
        card(20002, "ADH1", stage="binding"),
        card(20003, "ADH1"),
    ]
    store.set("model:scientist:" + cards[2]["entry_id"], False, 0, {}, "fixture-admin")
    first = page(library)
    assert first["models"][0]["entry_id"] == wanted and first["matching_entries"] == 10001
    store.set("model:scientist:" + wanted, False, 0, {}, "fixture-admin")
    filtered = page(library)
    assert filtered["matching_entries"] == 10000
    assert wanted not in [row["entry_id"] for row in filtered["models"]]


def test_equal_relevance_has_stable_pagination_despite_metadata_enumeration(catalog):
    cards, _, library = catalog
    cards[:] = [card(3, "ADH1"), card(1, "ADH1"), card(2, "ADH1")]
    first = page(library, limit=1)
    cards.reverse()
    second = page(library, offset=1, limit=1)
    assert first["models"][0]["entry_id"] == format(1, "064x")
    assert second["models"][0]["entry_id"] == format(2, "064x")


@pytest.mark.parametrize("query", [
    "predict ADH1 expression", "ADH1 expression prediction", "ADH1 丰度预测",
])
def test_archive_matches_prediction_intent_without_exposing_a_callable_model(catalog, query):
    cards, store, library = catalog
    wanted = cards[-1]["entry_id"]
    direct = page(library, query=query)
    assert direct["models"][0]["entry_id"] == wanted and direct["matching_entries"] == 10001
    public = discovery.search(query, kind="models", stage="expression", limit=2, tool_catalog=[])
    assert public["results"] == [] and public["matching_results"] == 0
    store.set("model:scientist:" + wanted, False, 0, {}, "fixture-admin")
    filtered = page(library, query=query)
    assert filtered["matching_entries"] == 10000
    assert wanted not in [row["entry_id"] for row in filtered["models"]]


@pytest.mark.parametrize("query", ["ADH1 docking", "ADH1 training", "ADH1 structure"])
def test_prediction_description_does_not_supply_other_capabilities(catalog, query):
    _, _, library = catalog
    assert evolved.list_models(library, query=query)["matching_entries"] == 0
    result = discovery.search(query, kind="models", stage="expression", tool_catalog=[])
    assert result["matching_results"] == 0
