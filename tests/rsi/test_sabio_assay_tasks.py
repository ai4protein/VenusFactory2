"""Synthetic source and split checks; no network or research model acquisition."""

import copy
import hashlib
import json
import math
import random

import pytest

from agent.rsi import generated_assay, model_tasks
from agent.rsi import sabio_assay_encoder as encoder
from agent.rsi import sabio_assay_tasks as tasks
from agent.rsi.engine.shared_model_bundle import admit
from agent.rsi.engine.storage import atomic_json


def observation(index=0, prop="Km"):
    return {
        "id": f"synthetic_{prop}_{index}",
        "protein": "SYNTH0",
        "organism": "Arabidopsis thaliana",
        "taxonomy_id": 3702,
        "publication_group": f"synthetic_publication_{index}",
        "ec_number": "1.1.1.37",
        "property": prop,
        "unit": tasks.UNITS[prop],
        "value": 2.0,
        "log10_value": math.log10(2),
        "associated_substrate": None if prop == "kcat" else {"id": 17, "name": "synthetic substrate"},
        "reaction": {"id": 19, "species": [{"role": "Substrate", "compound": {"id": 17}}]},
        "conditions": {
            "envvar_ph": {"start_value": 6.0, "end_value": 7.0},
            "envvar_temperature": {"start_value": 25.0, "unit": "°C"},
            "buffer": "synthetic 10 mM buffer",
            "envvars_other": [],
        },
    }


def entry(accession="SYNTH0", sequence="ACDEFGHIKLMNPQRSTVWY", taxon=3702):
    return {
        "primaryAccession": accession,
        "secondaryAccessions": [],
        "organism": {"taxonId": taxon, "scientificName": "Arabidopsis thaliana"},
        "sequence": {"value": sequence, "length": len(sequence), "md5": hashlib.md5(sequence.encode()).hexdigest()},
        "proteinDescription": {},
        "entryAudit": {"sequenceVersion": 2, "lastSequenceUpdateDate": "2000-01-01"},
    }


def test_encoder_is_bounded_label_free_and_preserves_intervals():
    row = observation()
    first = encoder.encode(row, "A" * 4096)
    assert len(first) == len(encoder.covariates()) == 205
    for value, column in zip(first, encoder.covariates(), strict=True):
        assert column["minimum"] <= value <= column["maximum"]
    changed = copy.deepcopy(row)
    changed.update(value=9999, log10_value=-99, target=999, protein="UNSEEN", id="another", publication_group="unseen")
    assert encoder.encode(changed, "A" * 4096) == first
    vector = dict(zip([x["name"] for x in encoder.covariates()], first, strict=True))
    assert (vector["ph_lower"], vector["ph_upper"], vector["ph_is_range"]) == (6, 7, 1)
    row["conditions"]["envvar_temperature"] = {}
    vector = dict(zip([x["name"] for x in encoder.covariates()], encoder.encode(row, "A" * 4096), strict=True))
    assert vector["temperature_missing"] == 1


@pytest.mark.parametrize(
    "value",
    [{"start_value": True}, {"start_value": 8, "end_value": 7}, {"start_value": float("nan")}, {"start_value": 15}],
)
def test_encoder_rejects_unresolved_conditions(value):
    with pytest.raises(ValueError):
        encoder.interval(value)


def test_encoder_reaction_and_associated_substrate_are_distinct():
    assert encoder.substrate_identity(observation(prop="kcat")) == "sabio_reaction:19"
    assert encoder.substrate_identity(observation(prop="Km")) == "sabio_compound:17"
    row = observation(prop="kcat")
    row["associated_substrate"] = {"id": 17}
    with pytest.raises(ValueError, match="invented"):
        encoder.encode(row, encoder.AA)
    row = observation()
    row["associated_substrate"]["id"] = 100
    with pytest.raises(ValueError, match="not a source substrate"):
        encoder.encode(row, encoder.AA)


@pytest.mark.parametrize(
    "change,reason",
    [
        ("fragment", "fragment_reference"),
        ("taxonomy", "source_reference_taxonomy_mismatch"),
        ("checksum", "sequence_checksum_mismatch"),
        ("length", "sequence_length_mismatch"),
    ],
)
def test_reference_failures_are_retained(change, reason):
    record = entry()
    if change == "fragment":
        record["proteinDescription"]["flag"] = "Fragment"
    if change == "taxonomy":
        record["organism"]["taxonId"] = 1
    if change == "checksum":
        record["sequence"]["md5"] = "0" * 32
    if change == "length":
        record["sequence"]["length"] += 1
    accepted, exclusions = tasks.bind_sequences([observation()], [{"results": [record]}])
    assert not accepted and exclusions == [{"protein": "SYNTH0", "reason": reason}]


def test_reference_subspecies_requires_official_ancestor_and_preserves_alias():
    row = observation()
    record = entry(accession="PRIMARY", taxon=1234)
    record["secondaryAccessions"] = [row["protein"]]
    accepted, excluded = tasks.bind_sequences([row], [{"results": [record]}])
    assert not accepted and excluded
    taxonomy = {"taxonId": 1234, "scientificName": record["organism"]["scientificName"], "parent": {"taxonId": 3702}}
    accepted, excluded = tasks.bind_sequences([row], [{"results": [record]}], [taxonomy])
    assert not excluded
    assert accepted[row["protein"]]["primary_accession"] == "PRIMARY"
    assert accepted[row["protein"]]["taxonomy_binding"] == "verified_taxonomic_ancestor"
    assert accepted[row["protein"]]["sequence_version"] == 2


def test_components_join_publications_sequence_aliases_and_similarity_edges():
    rows = [observation(i) for i in range(4)]
    for i, row in enumerate(rows):
        row["protein"] = f"SYNTH{i}"
    rows[1]["publication_group"] = rows[0]["publication_group"]
    references = {f"SYNTH{i}": {"sequence_sha256": str(i)} for i in range(4)}
    references["SYNTH2"] = references["SYNTH1"]
    groups = tasks.components(rows, references, [["SYNTH2", "SYNTH3"]])
    assert len(set(groups.values())) == 1
    assert tasks.components(list(reversed(rows)), references, [["SYNTH2", "SYNTH3"]]) == groups


def synthetic_sources(tmp_path):
    rng = random.Random(113)
    rows, records = [], []
    for group in range(14):
        protein = f"SYNTH{group}"
        sequence = "".join(rng.choice(encoder.AA) for _ in range(80 + group))
        records.append(entry(protein, sequence))
        for prop in tasks.PROPERTIES:
            for sample in range(6):
                row = observation(group * 100 + sample, prop)
                row.update(
                    protein=protein, publication_group=f"synthetic_publication_{group}", value=float(group + sample + 1)
                )
                row["log10_value"] = math.log10(row["value"])
                rows.append(row)
    source, sequences = tmp_path / "observations.json", tmp_path / "sequences.json"
    atomic_json(source, rows)
    atomic_json(sequences, {"results": records})
    return source, sequences, rows, records


def test_split_is_label_and_order_independent(tmp_path):
    _, _, rows, _ = synthetic_sources(tmp_path)
    groups = {r["id"]: r["protein"] for r in rows}
    assignment, summary = tasks.split_components(rows, groups)
    changed = copy.deepcopy(rows)
    for row in changed:
        row.update(value=9999, log10_value=-999)
    second, second_summary = tasks.split_components(list(reversed(changed)), groups)
    assert second == assignment and second_summary == summary
    with pytest.raises(ValueError, match="Insufficient"):
        tasks.split_components(rows, {r["id"]: "one" for r in rows})


def test_prepare_real_encoder_similarity_and_three_independent_tasks(tmp_path):
    source, sequences, rows, _ = synthetic_sources(tmp_path)
    output = tmp_path / "prepared"
    manifest = tasks.prepare(source, [sequences], [], output)
    assert manifest["new_admitted_tasks"] == 3 and manifest["new_autonomous_models"] == 0
    assert manifest["retained_observations"] == len(rows)
    definitions = [json.loads(model_tasks.bound_file(ref).read_text()) for ref in manifest["tasks"]]
    data = admit(definitions)
    for task in definitions:
        assert len(task["scope"]["covariates"]) == 205
        assert all("target" not in r for r in generated_assay.inference_inputs(data[task["id"]]["future"]))
        generated_assay.require_disjoint_parts(data[task["id"]], task["scope"])
    with pytest.raises(ValueError, match="fresh"):
        tasks.prepare(source, [sequences], [], output)


def test_similarity_connects_identical_references():
    refs = {"A": {"sequence": encoder.AA}, "B": {"sequence": encoder.AA}}
    report = tasks.similarity_evidence(refs)
    assert report["edges"] == [["A", "B"]]
    assert report["pairs"][0]["identity"] == report["pairs"][0]["minimum_coverage"] == 1
