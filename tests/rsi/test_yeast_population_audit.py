import copy
import csv
import hashlib
import zipfile

import pytest

from agent.rsi.yeast_population_audit import MEMBER, Q_FIELDS, audit_numeric


@pytest.fixture
def source_fixture(tmp_path):
    metadata = tmp_path / "metadata.tsv"
    rows, metadata_rows = [], []
    for batch in ("1", "2", "3"):
        for number in range(2):
            run = f"qc_{batch}_{number}"
            metadata_rows.append([run + ".rawIdx.Zip", "QC-" + str(number), batch])
            for pc, peptide, amount in [("P1", "PEPTIDEA", 100), ("P2", "PEPTIDEB", 200)]:
                rows.append(
                    {
                        "File.Name": "D:\\measurements\\" + run + ".rawIdx.wiff.dia",
                        "Genes": "",
                        "Protein.Ids": "P00001",
                        "Proteotypic": "1",
                        "Precursor.Id": pc,
                        "Stripped.Sequence": peptide,
                        "Precursor.Quantity": str(amount * int(batch)),
                        **dict.fromkeys(Q_FIELDS, "0.001"),
                    }
                )
    metadata_rows.extend([["a.rawIdx.Zip", "AAA", "1"], ["b.rawIdx.Zip", "SACE.ABC", "2"]])
    for run, quantity in [("a", 200), ("b", 800)]:
        for pc, peptide, multiple in [("P1", "PEPTIDEA", 1), ("P2", "PEPTIDEB", 2)]:
            rows.append(
                {
                    "File.Name": run + ".rawIdx.wiff.dia",
                    "Genes": "",
                    "Protein.Ids": "P00001",
                    "Proteotypic": "1",
                    "Precursor.Id": pc,
                    "Stripped.Sequence": peptide,
                    "Precursor.Quantity": str(quantity * multiple),
                    **dict.fromkeys(Q_FIELDS, "0.001"),
                }
            )
    with metadata.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["File.Name.Short", "Strain", "Batches"])
        writer.writerows(metadata_rows)
    table, archive = tmp_path / MEMBER, tmp_path / "report.zip"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(table, MEMBER)
    annotation, accessions = tmp_path / "genes.csv", tmp_path / "accessions.tsv"
    annotation.write_text("systematic_name,Gene\nYAL001C,TFC3\n")
    accessions.write_text("Entry\tGene Names (ordered locus)\tOrganism (ID)\nP00001\tYAL001C\t559292\n")
    panel = {
        hashlib.sha256(pc.encode()).hexdigest(): {
            "gene": "YAL001C",
            "peptide_sha256": hashlib.sha256(peptide.encode()).hexdigest(),
            "medians": {b: amount * int(b) for b in ("1", "2", "3")},
            "qc_observations": dict.fromkeys(("1", "2", "3"), 2),
            "qc_cv": dict.fromkeys(("1", "2", "3"), 0.0),
        }
        for pc, peptide, amount in [("P1", "PEPTIDEA", 100), ("P2", "PEPTIDEB", 200)]
    }
    genes = {
        "YAL001C": {
            "observations": {
                sample: {
                    "target": target,
                    "runs": [{"run": run, "batch": batch, "target": target, "distinct_peptides": 2, "precursors": 2}],
                }
                for sample, run, batch, target in [("AAA", "a", "1", 1.0), ("SACE_ABC", "b", "2", 2.0)]
            },
        }
    }
    return archive, metadata, annotation, accessions, panel, genes, {"AAA", "SACE_ABC"}


def test_independent_replay_matches_known_raw_arithmetic(source_fixture):
    result = audit_numeric(*source_fixture)
    assert result["observations"] == 2 and result["precursors"] == 2 and result["genes"] == 1
    assert result["qc_rows"]["rows_read"] == 16
    assert result["max_target_absolute_difference"] < 1e-12


def test_saved_qc_and_target_tampering_are_detected(source_fixture):
    arguments = list(source_fixture)
    wrong_panel = copy.deepcopy(arguments[4])
    first = next(iter(wrong_panel))
    wrong_panel[first]["medians"]["1"] *= 1.01
    arguments[4] = wrong_panel
    with pytest.raises(ValueError, match="QC arithmetic"):
        audit_numeric(*arguments)
    arguments = list(source_fixture)
    wrong_genes = copy.deepcopy(arguments[5])
    wrong_genes["YAL001C"]["observations"]["AAA"]["target"] += 0.01
    arguments[5] = wrong_genes
    with pytest.raises(ValueError, match="target differs"):
        audit_numeric(*arguments)


def test_missing_observation_and_wrong_taxon_are_detected(source_fixture):
    arguments = list(source_fixture)
    missing = copy.deepcopy(arguments[5])
    del missing["YAL001C"]["observations"]["AAA"]
    arguments[5] = missing
    with pytest.raises(ValueError, match="membership differs"):
        audit_numeric(*arguments)
    accessions = source_fixture[3]
    accessions.write_text(accessions.read_text().replace("559292", "9606"))
    with pytest.raises(ValueError, match="non-yeast"):
        audit_numeric(*source_fixture)
