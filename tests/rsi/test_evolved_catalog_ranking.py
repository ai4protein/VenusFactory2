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


def pages(library, *, offset=0, limit=2):
    direct = evolved.list_models(
        library, query="ADH1 expression", stage="expression", limit=limit, offset=offset
    )
    combined = discovery.search(
        "ADH1 expression", kind="models", stage="expression",
        limit=limit, offset=offset, tool_catalog=[],
    )
    return direct, combined


def test_named_protein_is_not_buried_by_ten_thousand_description_matches(catalog):
    cards, _, library = catalog
    direct, combined = pages(library)
    wanted = cards[-1]["entry_id"]
    assert direct["models"][0]["entry_id"] == wanted
    assert combined["results"][0]["entry_id"] == wanted
    assert direct["matching_entries"] == combined["matching_results"] == 10001
    assert direct["next_offset"] == combined["next_offset"] == 2
    assert len(direct["models"]) == len(combined["results"]) == 2
    assert all("scope" not in row and "inputSchema" not in row for row in direct["models"])
    next_direct, next_combined = pages(library, offset=2)
    for direct_page, combined_page in [(direct, combined), (next_direct, next_combined)]:
        assert [x["entry_id"] for x in direct_page["models"]] == [
            x["entry_id"] for x in combined_page["results"]
        ]
    assert not {x["entry_id"] for x in direct["models"]} & {
        x["entry_id"] for x in next_direct["models"]
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
    direct, combined = pages(library)
    assert direct["models"][0]["entry_id"] == combined["results"][0]["entry_id"] == wanted
    assert direct["matching_entries"] == combined["matching_results"] == 10001
    store.set("model:scientist:" + wanted, False, 0, {}, "fixture-admin")
    direct, combined = pages(library)
    assert direct["matching_entries"] == combined["matching_results"] == 10000
    assert wanted not in [x["entry_id"] for x in direct["models"] + combined["results"]]


def test_equal_relevance_has_stable_pagination_despite_metadata_enumeration(catalog):
    cards, _, library = catalog
    cards[:] = [card(3, "ADH1"), card(1, "ADH1"), card(2, "ADH1")]
    first, combined = pages(library, limit=1)
    cards.reverse()
    second, next_combined = pages(library, offset=1, limit=1)
    assert first["models"][0]["entry_id"] == combined["results"][0]["entry_id"] == format(1, "064x")
    assert second["models"][0]["entry_id"] == next_combined["results"][0]["entry_id"] == format(2, "064x")
