import copy
import hashlib
import json

import pytest

from agent.rsi import maize_population_source as source


@pytest.fixture
def tables():
    pheader = [
        "Uniprot_Accession",
        "Ensembl_GID",
        "LINE1 (NSS)",
        "Coverage",
        "#Unique Peptides",
        "#Peptides",
        "LINE2 (Mixed)",
        "Coverage",
        "#Unique Peptides",
        "#Peptides",
    ]
    rheader = ["Ensembl_GID", "LINE2 (Mixed)", "LINE1 (NSS)"]
    proteins = [["P1", "GENE1", 2, 0.2, 2, 3, 0.5, 0.3, 3, 4], ["P2", "GENE2", 1, 0.2, 1, 2, "NA", "NA", "NA", "NA"]]
    rna = [["GENE2", 5, 4], ["GENE1", -2, 3]]
    return pheader, proteins, rheader, rna


def test_exact_join_and_cell_provenance(tables):
    genes, counts = source.pair_tables(*tables)
    first, second = genes["GENE1"]["observations"]
    assert (first["sample"], first["author_reported_protein"], first["author_reported_rna"]) == ("LINE1", 2, 3)
    assert (second["sample"], second["author_reported_protein"], second["author_reported_rna"]) == ("LINE2", 0.5, -2)
    assert first["source_cells"] == {"protein_ratio": [9, 3], "unique_peptides": [9, 5], "rna": [4, 3]}
    assert counts["excluded_pairs"] == {"fewer_than_two_unique_peptides": 1, "missing_protein_ratio": 1}
    assert counts["negative_rna_values_retained"] == 1
    assert counts["measured_pairs_retained"] == 2
    assert counts["admitted_tasks"] == counts["autonomous_models"] == 0


@pytest.mark.parametrize("value", [float("inf"), float("nan"), "2", True])
def test_invalid_protein_measurement_fails(tables, value):
    tables[1][0][2] = value
    with pytest.raises(ValueError):
        source.pair_tables(*tables)


@pytest.mark.parametrize("value", [0, -1])
def test_nonpositive_source_values_preserved_without_acquisition(tables, value):
    tables[1][0][2] = value
    genes, summary = source.pair_tables(*tables)
    assert genes["GENE1"]["observations"][0]["author_reported_protein"] == value
    assert summary["nonpositive_protein_values_observed_in_paired_samples"] == 1
    assert summary["nonpositive_protein_values_retained"] == 1
    assert source.POLICY["ready_for_acquisition"] is False
    assert source.POLICY["protein_transformation"] == "unresolved_nonpositive_reported_ratios"
    assert summary["admitted_tasks"] == 0


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_peptide_count_fails(tables, value):
    tables[1][0][4] = value
    with pytest.raises(ValueError):
        source.pair_tables(*tables)


@pytest.mark.parametrize(
    "change", ["duplicate_gene", "duplicate_sample", "foreign_gene", "foreign_sample", "population", "quality_header"]
)
def test_ambiguous_or_inconsistent_join_fails(tables, change):
    if change == "duplicate_gene":
        tables[3].append(copy.deepcopy(tables[3][0]))
    elif change == "duplicate_sample":
        tables[2][2] = tables[2][1]
    elif change == "foreign_gene":
        tables[3][0][0] = "OTHER"
    elif change == "foreign_sample":
        tables[2][2] = "OTHER (NSS)"
    elif change == "population":
        tables[2][2] = "LINE1 (TST)"
    else:
        tables[0][4] = "#Peptides"
    with pytest.raises(ValueError):
        source.pair_tables(*tables)


def test_missing_rna_not_imputed(tables):
    tables[3][1][1] = "NA"
    genes, counts = source.pair_tables(*tables)
    assert len(genes["GENE1"]["observations"]) == 1
    assert counts["excluded_pairs"]["missing_rna"] == 1


def test_export_hash_binding_and_acquisition_block(tmp_path, tables, monkeypatch):
    import openpyxl

    inputs = tmp_path / "inputs"
    inputs.mkdir()
    hashes = {}
    for name, header_row, header, rows in [("p.xlsx", 8, tables[0], tables[1]), ("r.xlsx", 2, tables[2], tables[3])]:
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "Sheet1"
        for _ in range(header_row - 1):
            sheet.append(["metadata"])
        sheet.append(header)
        for row in rows:
            sheet.append(row)
        path = inputs / name
        book.save(path)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(source, "SOURCE_FILES", hashes)
    output = tmp_path / "export"
    manifest = source.export_source(inputs, output)
    assert manifest["policy"]["ready_for_acquisition"] is False
    assert manifest["policy"]["rna_transformation"] == "unresolved"
    for gene in manifest["genes"].values():
        assert source.digest(output / gene["path"]) == gene["sha256"]
    saved = json.loads((output / "manifest.json").read_text())
    assert saved == manifest
    with pytest.raises(ValueError, match="already exists"):
        source.export_source(inputs, output)
    (inputs / "p.xlsx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        source.export_source(inputs, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()
