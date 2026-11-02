from copy import deepcopy

import pytest
import test_scientist_model_library as fixtures

from agent.rsi.model_scope_inventory import canonical_scope, inventory_libraries, summarize_catalogs
from agent.rsi.storage import atomic_json

storage = fixtures.storage


def key():
    return {
        "plugin_id": "generated_context.v1",
        "scope": {
            "protein": "Protein A",
            "target": "abundance",
            "unit": "log2",
            "assay": "study",
            "context": {"growth": "respiration"},
        },
        "program_sha256": "a" * 64,
        "training_sha256": "b" * 64,
    }


def catalog(name, keys):
    return {
        "library": name,
        "catalog": {
            "indexed_entries": len(keys),
            "entries": [{"entry_id": str(i), "key": k} for i, k in enumerate(keys)],
        },
    }


def test_versions_partitions_and_interface_do_not_inflate_coverage():
    original = key()
    version = deepcopy(original)
    version.update(program_sha256="c" * 64, training_sha256="d" * 64, plugin_id="generated_context.v2")
    version["scope"]["protein"] = "  PROTEIN   a "
    r = summarize_catalogs([catalog("one", [original, version]), catalog("two", [deepcopy(original)])])
    assert (r["registered_entries"], r["registered_scopes"], r["repeated_scope_registrations"]) == (3, 1, 2)
    assert r["scopes_in_multiple_libraries"] == 1 and r["distinct_source_programs"] == 2
    assert r["currently_reverified_scopes"] is None and not r["completion_claim"]


@pytest.mark.parametrize(
    "field,value", [("protein", "Protein B"), ("target", "solubility"), ("context", {"growth": "fermentation"})]
)
def test_distinct_biological_scope_adds_coverage(field, value):
    original = key()
    other = deepcopy(original)
    other["scope"][field] = value
    assert summarize_catalogs([catalog("one", [original, other])])["registered_scopes"] == 2


def test_scope_matches_existing_campaign_rule():
    from agent.rsi.shared_model_campaign import task_scope

    original = key()
    assert canonical_scope(original) == task_scope({"plugin_id": original["plugin_id"], "scope": original["scope"]})


def test_duplicate_library_and_bad_metadata_rejected():
    c = catalog("same", [key()])
    with pytest.raises(ValueError, match="distinct"):
        summarize_catalogs([c, deepcopy(c)])
    c["catalog"]["indexed_entries"] += 1
    with pytest.raises(ValueError, match="count"):
        summarize_catalogs([c])


def test_registry_metadata_checked_without_native_replay(storage):
    library, goal, verified, calls = storage
    library.publish(goal, verified["packet"]["model_id"])
    calls.clear()
    r = inventory_libraries([str(library.root)])
    assert r["registered_scopes"] == 1 and not calls and r["catalog_metadata_stable_during_inventory"]
    atomic_json(next((library.root / "index").glob("*.json")), {"entry_id": "0" * 64})
    with pytest.raises((ValueError, FileNotFoundError)):
        inventory_libraries([str(library.root)])


def test_missing_library_and_nonpositive_target_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        inventory_libraries([str(tmp_path / "missing")])
    with pytest.raises(ValueError, match="Positive"):
        summarize_catalogs([], target_scopes=0)
