"""Synthetic native-ingestion tests; no model training or external calls."""
import copy
import csv
import io
import json

import pytest

from agent.harness.research_data_store import ResearchDataStore
from agent.harness.research_dataset import prepare_dataset
from agent.rsi.datasets.skempi_protocol import ALIGNMENT, SCHEMA
from agent.rsi.datasets.skempi_source import canonical, sha256
from agent.rsi.datasets.skempi_task import eligible_rows, export_protocol, grouped_splits, model_inputs


def rows():
    aa = "ACDEFGHIKLMNPQRSTVWY"
    result = []
    for i in range(500):
        group = i // 10
        suffix = aa[i // 400] + aa[(i // 20) % 20] + aa[i % 20]
        result.append({"id": "fixture-" + str(i), "source_line": i + 2, "complex_id": f"{group:04d}_A_B",
                       "exclusion_group": sha256(str(group).encode()), "duplicate_fingerprint": sha256(str(i).encode()),
                       "sequence_mapping_status": "verified_observed_residues",
                       "partner_sequences": {"A": {"observed_sequence": "A" * 25 + suffix},
                                             "B": {"observed_sequence": "G" * 25 + suffix}},
                       "partner_chains": ["A", "B"],
                       "mutations_pdb": [{"chain": "A", "wild_type": "A", "mutant": "C"}],
                       "mutation_mapping": [{"sequence_index": 0}],
                       "temperature": {"kelvin": 298},
                       "raw": {"Method": "SPR", "Reference": str(group)},
                       "derived_ddg": {"value": float(i % 7)}})
    return result


def audited(tmp_path):
    values = rows()
    body = "".join(canonical(row) + "\n" for row in values).encode()
    (tmp_path / "observations.jsonl").write_bytes(body)
    report = {"schema": SCHEMA, "complete": True, "method": ALIGNMENT, "observations_sha256": sha256(body)}
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report))
    return path, sha256(path.read_bytes())


def test_whole_groups_are_label_independent_and_never_truncated():
    data = rows()
    chosen = grouped_splits(data)
    changed = copy.deepcopy(list(reversed(data)))
    for item in changed:
        item["derived_ddg"]["value"] = -999
    again = grouped_splits(changed)
    assert {n: [r["id"] for r in v] for n, v in chosen.items()} == {n: [r["id"] for r in v] for n, v in again.items()}
    assert sum(map(len, chosen.values())) == len(data)
    owners = {}
    for name, values in chosen.items():
        if name != "train":
            assert 32 <= len(values) <= 512
        for item in values:
            assert owners.setdefault(item["exclusion_group"], name) == name
    for item in data:
        item["exclusion_group"] = "one-linked-biological-component"
    with pytest.raises(ValueError, match="never split groups"):
        grouped_splits(data)


def test_eligibility_and_mutant_sequences_do_not_leak_affinities():
    data = rows()
    data[0]["sequence_mapping_status"] = "requires_review"
    data[1]["derived_ddg"]["value"] = None
    data[2]["partner_sequences"]["A"]["observed_sequence"] = "AAA"
    data[4]["duplicate_fingerprint"] = data[3]["duplicate_fingerprint"]
    eligible, reasons = eligible_rows(data)
    assert len(eligible) == 496 and set(reasons) == {
        "unverified_mapping", "nonexact_or_unmeasured_temperature", "short_partner_chain", "exact_source_duplicate"}
    inputs = model_inputs(eligible[0])
    assert json.loads(inputs["wild_type_chains"])["A"][0] == "A"
    assert json.loads(inputs["mutant_chains"])["A"][0] == "C"
    assert "target" not in inputs and "affinity" not in canonical(inputs).lower()
    damaged = copy.deepcopy(eligible[0])
    damaged["mutations_pdb"][0]["wild_type"] = "W"
    with pytest.raises(ValueError, match="verified WT"):
        model_inputs(damaged)


def test_native_handoff_keeps_validation_and_test_labels_host_only(tmp_path):
    path, checksum = audited(tmp_path)
    protocol = export_protocol(path, checksum, tmp_path / "protocol")
    assert protocol["state"] == "prepared_not_trained" and protocol["model_task_count"] == 1
    assert protocol["trained_models"] == protocol["external_model_requests"] == 0
    data = (tmp_path / "protocol/dataset.csv").read_bytes()
    all_rows = list(csv.DictReader(io.StringIO(data.decode())))
    work = tmp_path / "workspace"
    work.mkdir()
    store = ResearchDataStore(tmp_path / "private", work)
    store.ingest("inputs/binding_ddg.csv", data, protocol["artifacts"]["dataset.csv"]["sha256"])
    prepared = prepare_dataset(work, data_store=store, **protocol["native_prepare"])
    visible = list(csv.DictReader(io.StringIO((work / "inputs/binding_ddg.csv").read_text())))
    assert {r["id"] for r in visible} == {r["id"] for r in all_rows if r["split"] == "train"}
    frozen = json.loads(store.load_evaluation(prepared["evaluation_file"]))
    for name, values in frozen["splits"].items():
        assert len(values) == protocol["cohorts"][name]["rows"]
        assert all(set(r["inputs"]) == set(protocol["native_prepare"]["input_columns"]) for r in values)
    handle = json.loads((work / prepared["evaluation_file"]).read_text())
    assert set(handle) == {"schema", "id"} and prepared["evaluation_access"] == "host_reference"
    again = export_protocol(path, checksum, tmp_path / "again")
    assert again == protocol
    assert (tmp_path / "again/dataset.csv").read_bytes() == data


@pytest.mark.parametrize("asset", ["audit.json", "observations.jsonl"])
def test_modified_audit_or_data_cannot_prepare_task(tmp_path, asset):
    path, checksum = audited(tmp_path)
    target = tmp_path / asset
    target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        export_protocol(path, checksum, tmp_path / "out")
    assert not (tmp_path / "out").exists()
