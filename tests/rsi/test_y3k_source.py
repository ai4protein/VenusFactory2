import copy

import numpy as np
import pytest

from agent.rsi.datasets.y3k_source import FEATURES, parse_condition, protein_index, strain_index


@pytest.fixture
def source_rows():
    strains = [
        ["YER069W", "ARG5,6", 1, "Q01217", "NAGS"],
        ["YOR286W", "RDL2 (AIM42)", 2, "Q08742", "TSTD1"],
        ["YDR393W", "SHE9 (MDM33)", 3, "Q04172", "None"],
        ["None (WT)", "None (WT)", "n.a. (WT)", "n.a. (WT)", "n.a. (WT)"],
    ]
    proteins = [["PROTEIN", "Example", "ACO1", "YLR304c", "P19414"]]
    header = ["Molecule Name", "Molecule Type", "∆ARG5", "∆RDL2", "∆YDR393W"]
    rows = [[name, "Metabolite", 0.0, -1.0, None] for name in FEATURES]
    rows += [["ACO1 (YLR304C)", "Protein", None, -2.0, 0.0]]
    return strains, proteins, header, rows


def parse(source):
    strains, proteins, header, rows = source
    index, _ = strain_index(strains)
    accepted, excluded = protein_index(proteins)
    return parse_condition(header, rows, index, accepted, excluded)


def test_exact_aliases_and_preserved_missing_signed_measurements(source_rows):
    genes, proteins, x, y, witness = parse(source_rows)
    assert genes == ["YER069W", "YOR286W", "YDR393W"]
    assert proteins == ["P19414"]
    assert x.shape == (3, 19) and y.shape == (3, 1)
    assert (x[0] == 0).all() and (x[1] == -1).all() and np.isnan(x[2]).all()
    assert np.isnan(y[0, 0]) and y[1, 0] == -2 and y[2, 0] == 0
    assert witness["columns"][2]["source_label"] == "∆YDR393W"
    assert witness["features"]["Alanine"]["row"] == 2
    assert witness["proteins"]["P19414"]["row"] == 21


@pytest.mark.parametrize("value", [True, "1.2", "=A2", "#N/A", float("inf"), float("nan")])
def test_invalid_measurements_are_not_silently_imputed(source_rows, value):
    source_rows[3][0][2] = value
    with pytest.raises(ValueError, match="Invalid measurement"):
        parse(source_rows)


def test_duplicate_canonical_genotype_is_rejected(source_rows):
    source_rows[2][3] = "∆YER069W"
    with pytest.raises(ValueError, match="Repeated canonical"):
        parse(source_rows)


def test_unknown_alias_is_rejected(source_rows):
    source_rows[2][2] = "∆ARG6"
    with pytest.raises(ValueError, match="Unresolved deletion"):
        parse(source_rows)


def test_alias_requires_exact_source_locus_and_label(source_rows):
    source_rows[0][0][1] = "ARG5-like"
    with pytest.raises(ValueError, match="Unresolved deletion"):
        parse(source_rows)


@pytest.mark.parametrize("operation", ["missing", "duplicate"])
def test_feature_panel_is_exact(source_rows, operation):
    if operation == "missing":
        source_rows[3].pop(0)
    else:
        source_rows[3].append(copy.deepcopy(source_rows[3][0]))
    with pytest.raises(ValueError, match="amino-acid measurement|Repeated requested metabolite"):
        parse(source_rows)


def test_unrequested_duplicate_metabolites_do_not_alias_requested_features(source_rows):
    source_rows[3].extend([["Glyceric acid", "Metabolite", 1, 2, 3]] * 2)
    assert parse(source_rows)[2].shape == (3, 19)


def test_ambiguous_isoform_loci_and_multiple_accessions_are_excluded(source_rows):
    source_rows[1].extend(
        [
            ["PROTEIN", "Isoform 1", "LEU4", "YNL104C", "P06208;P06208-2"],
            ["PROTEIN", "Isoform 2", "LEU4", "YNL104c", "P06208"],
            ["PROTEIN", "Unresolved", "TEST", "YLR305C", "P00001;P00002"],
        ]
    )
    source_rows[3].extend(
        [
            ["LEU4 (YNL104c)", "Protein", 1, 2, 3],
            ["TEST (YLR305C)", "Protein", 1, 2, 3],
        ]
    )
    _, ids, _, y, witness = parse(source_rows)
    assert ids == ["P19414"] and y.shape == (3, 1)
    assert {x["reason"] for x in witness["excluded_protein_rows"]} == {
        "multiple_source_protein_records_for_locus",
        "not_a_single_resolved_accession",
    }


def test_protein_gene_and_locus_must_both_match(source_rows):
    source_rows[3][-1][0] = "Wrong (YLR304C)"
    with pytest.raises(ValueError, match="does not match"):
        parse(source_rows)


def test_repeated_target_is_rejected(source_rows):
    source_rows[3].append(copy.deepcopy(source_rows[3][-1]))
    with pytest.raises(ValueError, match="Repeated resolved protein"):
        parse(source_rows)


def test_duplicate_accession_cannot_create_extra_protein(source_rows):
    source_rows[1].append(["PROTEIN", "Other", "OTHER", "YLR305C", "P19414"])
    with pytest.raises(ValueError, match="Accession assigned"):
        parse(source_rows)


@pytest.mark.parametrize("locus,gene,accession", [("Q0250", "COX2", "P00410"), ("R0020c", "REP1", "P03871")])
def test_documented_mitochondrial_and_plasmid_loci(source_rows, locus, gene, accession):
    source_rows[1].append(["PROTEIN", "Other", gene, locus, accession])
    source_rows[3].append([f"{gene} ({locus})", "Protein", 1, 2, 3])
    assert accession in parse(source_rows)[1]


@pytest.mark.parametrize("locus", ["YDR385wYOR133w", "YMR125w2", "YHR215w;YAR071w"])
def test_unresolved_locus_groups_are_retained_as_exclusions(source_rows, locus):
    source_rows[1].append(["PROTEIN", "Other", "OTHER", locus, "P00001"])
    source_rows[3].append([f"OTHER ({locus})", "Protein", 1, 2, 3])
    _, ids, _, _, witness = parse(source_rows)
    assert ids == ["P19414"]
    assert witness["excluded_protein_rows"][-1]["reason"] == "not_a_single_resolved_locus"


def test_excel_date_gene_label_requires_independent_identity_match():
    from datetime import datetime

    rows = [["PROTEIN", "Mitochondrial intermediate peptidase", datetime(2016, 10, 1), "YKL134C", "P35999"]]
    _, metadata = strain_index([["YKL134C", "OCT1", 1, "P35999", "MIPEP"]])
    accepted, excluded = protein_index(rows, metadata)
    assert not excluded
    assert accepted["YKL134C"]["gene"] == "OCT1"
    assert accepted["YKL134C"]["original_gene_value"] == "2016-10-01T00:00:00"
    assert accepted["YKL134C"]["gene_label_repair"]["strain_table_row"] == 2
    metadata["YKL134C"]["accession"] = "P00001"
    accepted, excluded = protein_index(rows, metadata)
    assert not accepted and excluded["YKL134C"]["reason"] == "unresolved_gene_label"
    accepted, excluded = protein_index(rows)
    assert not accepted and excluded["YKL134C"]["reason"] == "unresolved_gene_label"
