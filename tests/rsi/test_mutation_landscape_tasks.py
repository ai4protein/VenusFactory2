"""Scientific data contracts; synthetic observations are never publications."""

import csv
import itertools
import json
from copy import deepcopy

import pytest

from agent.rsi.datasets.mutation_landscape_tasks import (
    gfp_parents,
    gfp_records,
    make_record,
    partition_records,
    reconstruct,
    trpb_records,
)

PARENT = "MACDEFGHIKLMNPQRSTVWY"


def test_zero_based_and_original_av_coordinates_check_wild_type_residues():
    native, distance = reconstruct(PARENT, "A1C:D3E")
    original, other = reconstruct(PARENT, "SA0C:SD2E", av_format=True)
    assert native == original == "MCCEEFGHIKLMNPQRSTVWY"
    assert distance == other == 2
    assert reconstruct(PARENT, "wt") == (PARENT, 0)


@pytest.mark.parametrize("genotype", ["A0C", "A1C:A1D", "A1A", "A100C", "A1*", "A1X", "A-1C", ""])
def test_invalid_substitutions_cannot_be_silently_applied(genotype):
    with pytest.raises(ValueError):
        reconstruct(PARENT, genotype)


def test_parent_extraction_does_not_execute_notebook(tmp_path):
    source = "raise AssertionError('Never execute author code')\n"
    source += "\n".join(f"wt_seqs[{name!r}] = {PARENT + '*'!r}" for name in ["avGFP", "amacGFP", "cgreGFP", "ppluGFP"])
    path = tmp_path / "source.ipynb"
    path.write_text(json.dumps({"cells": [{"cell_type": "code", "source": [source]}]}))
    assert set(gfp_parents(path).values()) == {PARENT}


def test_gfp_uses_one_log_transform_and_preserves_existing_av_log_scale(tmp_path):
    native, av = tmp_path / "native.tsv", tmp_path / "av.tsv"
    native.write_text("\taa_genotype_native\tgene\treplicates_mean_brightness\n7\tA1C\tamacGFP\t100\n")
    av.write_text("aaMutations\tmedianBrightness\nSA0C\t3.5\n")
    records, excluded = gfp_records(native, av, {"avGFP": PARENT, "amacGFP": PARENT})
    assert [row["target"] for row in records] == [2.0, 3.5]
    assert records[0]["sequence"] == records[1]["sequence"]
    assert not excluded


def trpb_tables(tmp_path, replicas=1):
    normalized = [-0.2] if replicas == 1 else [-0.2, 0.0]
    fitness = sum(normalized) / replicas
    summary = {"AAs": "ACD", "mu_1": 0.3, "InputCount_1": 100, "mu_1-bg/max": normalized[0]}
    if replicas == 2:
        summary.update({"mu_2": 0.4, "InputCount_2": 120, "mu_2-bg/max": normalized[1], "avg_mu-bg/max": fitness})
    simple = {"AAs": "ACD", "AA1": "A", "AA2": "C", "AA3": "D", "# Stop": 0, "fitness": fitness}
    return summary, simple


def write_table(path, row):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def read_trpb(tmp_path, summary, simple):
    a, b = tmp_path / "summary.csv", tmp_path / "simple.csv"
    write_table(a, summary)
    write_table(b, simple)
    libraries = {"A": {"positions": {1: 2, 2: 3, 3: 4}}}
    return trpb_records(b, a, PARENT, libraries, "A")


@pytest.mark.parametrize("replicas", [1, 2])
def test_trpb_retains_measured_negative_fitness_and_exact_replicate_count(tmp_path, replicas):
    summary, simple = trpb_tables(tmp_path, replicas)
    records, audit = read_trpb(tmp_path, summary, simple)
    assert records[0]["target"] == simple["fitness"] < 0
    assert records[0]["sequence"] == PARENT and records[0]["mutation_count"] == 0
    assert audit["observed_replicates"] == replicas


@pytest.mark.parametrize("attack", ["fitness", "mean", "tuple", "missing_source", "stop_count"])
def test_trpb_rejects_changed_or_unmeasured_values(tmp_path, attack):
    summary, simple = trpb_tables(tmp_path, 2)
    if attack == "fitness":
        simple["fitness"] = 10
    elif attack == "mean":
        summary["mu_2-bg/max"] = 2
    elif attack == "tuple":
        simple["AA2"] = "A"
    elif attack == "missing_source":
        simple["AAs"] = "CCC"
    else:
        simple["# Stop"] = 1
    with pytest.raises(ValueError):
        read_trpb(tmp_path, summary, simple)


@pytest.mark.parametrize("missing", ["", "nan", "0"])
def test_missing_read_support_is_excluded_instead_of_filled(tmp_path, missing):
    summary, simple = trpb_tables(tmp_path)
    summary["InputCount_1"] = missing
    records, audit = read_trpb(tmp_path, summary, simple)
    assert records == [] and audit["excluded"] == {"missing_measured_support": 1}


def observations():
    result = []
    for distance in [1, 2, 3]:
        for positions in itertools.combinations(range(20), distance):
            for letters in itertools.product("CDE", repeat=distance) if distance < 3 else ["CCC"]:
                sequence = list("A" * 20)
                for position, letter in zip(positions, letters, strict=True):
                    sequence[position] = letter
                result.append(
                    make_record(str(len(result)), "".join(sequence), distance, float(len(result) % 7), "fixture")
                )
    return result


def test_higher_order_holdout_and_caps_are_label_independent():
    records = observations()
    data, selected, counts = partition_records(records)
    changed = deepcopy(records)
    for row in changed:
        row["target"] = -row["target"] * 1000
    other, other_selected, other_counts = partition_records(changed)
    assert selected == other_selected and counts == other_counts
    assert {phase: len(rows) for phase, rows in data.items()} == {"training": 600, "validation": 250, "future": 250}
    assert all(row["mutation_count"] <= 2 for phase in ["training", "validation"] for row in selected[phase])
    assert all(row["mutation_count"] >= 3 for row in selected["future"])
    assert len({row["sequence"] for rows in data.values() for row in rows}) == 1100
    assert [row["id"] for row in other["future"]] == [row["id"] for row in data["future"]]


def test_published_indels_are_explicitly_excluded_from_substitution_tasks(tmp_path):
    native, av = tmp_path / "native.tsv", tmp_path / "av.tsv"
    native.write_text("\taa_genotype_native\tgene\treplicates_mean_brightness\n7\tA1.\tamacGFP\t100\n")
    av.write_text("aaMutations\tmedianBrightness\n")
    records, excluded = gfp_records(native, av, {"amacGFP": PARENT})
    assert records == []
    assert excluded == {"amacGFP:indel_outside_substitution_scope": 1}


def test_trpb_preserves_two_growth_replicates_with_one_published_input_column(tmp_path):
    summary, simple = trpb_tables(tmp_path, 2)
    del summary["InputCount_2"]
    records, audit = read_trpb(tmp_path, summary, simple)
    assert len(records) == 1
    assert audit["observed_replicates"] == 2
    assert audit["input_count_columns"] == ["InputCount_1"]
