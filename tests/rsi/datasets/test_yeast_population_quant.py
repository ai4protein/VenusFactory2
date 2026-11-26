import csv
import hashlib
import json
import zipfile
from collections import Counter

import pytest

from agent.rsi import yeast_population_quant as source


def row(precursor="P1", peptide="PEPA", quantity=100, gene="G1"):
    return {
        "File.Name": "run.rawIdx.wiff.dia",
        "Run": "run",
        "Genes": gene,
        "Protein.Ids": "",
        "Proteotypic": "1",
        "Precursor.Id": precursor,
        "Stripped.Sequence": peptide,
        "Precursor.Quantity": str(quantity),
        **dict.fromkeys(source.QUALITY_FIELDS, "0.001"),
    }


def fixture():
    runs = {}
    records = []
    for batch in ("1", "2", "3"):
        for injection in range(5):
            name = f"qc_{batch}_{injection}"
            runs[name] = {"qc": True, "sample": None, "batch": batch}
            records += [(name, row(quantity=100 * int(batch))), (name, row("P2", "PEPB", 200 * int(batch)))]
    runs.update(
        a={"qc": False, "sample": "AAA", "batch": "1"},
        a_repeat={"qc": False, "sample": "AAA", "batch": "2"},
        b={"qc": False, "sample": "BBB", "batch": "3"},
    )
    return runs, records, {"G1": "YAL001C"}


def test_qc_calibration_technical_runs_and_two_distinct_peptides():
    runs, records, aliases = fixture()
    panel = source.qc_panel(records, runs, aliases, Counter())
    measured = [
        ("a", row(quantity=200)),
        ("a", row("P2", "PEPB", 400)),
        ("a_repeat", row(quantity=800)),
        ("a_repeat", row("P2", "PEPB", 1600)),
        ("b", row(quantity=1200)),
        ("b", row("P2", "PEPB", 2400)),
    ]
    actual = source.quantify_samples(measured, runs, aliases, panel, {"AAA", "BBB"}, Counter())
    assert actual["YAL001C"]["AAA"]["target"] == pytest.approx(1.5)
    assert actual["YAL001C"]["BBB"]["target"] == pytest.approx(2.0)
    assert len(actual["YAL001C"]["AAA"]["runs"]) == 2
    one_peptide = [(run, r) for run, r in measured if r["Precursor.Id"] == "P1"]
    assert source.quantify_samples(one_peptide, runs, aliases, panel, {"AAA", "BBB"}, Counter()) == {}
    assert len(panel) == 2


def test_qc_panel_rejects_low_coverage_high_variation_and_non_qc_data():
    runs, records, aliases = fixture()
    missing = [(run, r) for run, r in records if not (run in {"qc_1_0", "qc_1_1"} and r["Precursor.Id"] == "P1")]
    assert len(source.qc_panel(missing, runs, aliases, Counter())) == 1
    noisy = [(run, dict(r, **{"Precursor.Quantity": "10000"}) if run == "qc_1_0" else r) for run, r in records]
    assert not source.qc_panel(noisy, runs, aliases, Counter())
    with pytest.raises(ValueError, match="Only QC"):
        source.qc_panel(records + [("a", row())], runs, aliases, Counter())


def test_conflicting_duplicate_signals_and_nonselected_samples_fail():
    runs, records, aliases = fixture()
    with pytest.raises(ValueError, match="Conflicting"):
        source.qc_panel(records + [(records[0][0], row(quantity=101))], runs, aliases, Counter())
    panel = source.qc_panel(records, runs, aliases, Counter())
    with pytest.raises(ValueError, match="Conflicting"):
        source.quantify_samples([("a", row()), ("a", row(quantity=101))], runs, aliases, panel, {"AAA"}, Counter())
    with pytest.raises(ValueError, match="Nonselected"):
        source.quantify_samples([("b", row())], runs, aliases, panel, {"AAA"}, Counter())


@pytest.mark.parametrize(
    "updates",
    [
        {"Genes": "G1;G2"},
        {"Genes": "unknown"},
        {"Proteotypic": "0"},
        {"Precursor.Quantity": "nan"},
        {"Precursor.Quantity": "0"},
        {"Precursor.Quantity": "-1"},
        {"Q.Value": "0.02"},
        {"Lib.PG.Q.Value": "-0.1"},
        {"Global.Q.Value": "inf"},
    ],
)
def test_invalid_signal_or_identity_is_excluded(updates):
    assert source.qualified(dict(row(), **updates), {"G1": "YAL001C"}) is None


def test_normalized_signal_is_ignored_and_holdout_targets_cannot_change_panel():
    runs, records, aliases = fixture()
    first = source.qc_panel(records, runs, aliases, Counter())
    second = source.qc_panel(
        [(name, dict(r, **{"Precursor.Normalised": "999999999"})) for name, r in records], runs, aliases, Counter()
    )
    assert first == second
    immutable = json.dumps(first, sort_keys=True)
    source.quantify_samples(
        [("b", row(quantity=1e100)), ("b", row("P2", "PEPB", 1e-100))], runs, aliases, first, {"BBB"}, Counter()
    )
    assert json.dumps(first, sort_keys=True) == immutable


def test_gene_aliases_and_rna_source_columns(tmp_path):
    annotation = tmp_path / "annotations.csv"
    annotation.write_text("systematic_name,Gene\nYAL001C,G1\nYAL002W,AMB\nYAL003W,AMB\n")
    aliases = source.gene_aliases(annotation)
    assert aliases["G1"] == "YAL001C" and "AMB" not in aliases
    rna = tmp_path / "rna.csv"
    rna.write_text("ORF,AAA,SACE_YBY,CCC\nYAL001C,1,2,nan\nYAL002W,7,8,9\n")
    assert source.rna_covariates(rna, {"YAL001C"}, {"AAA", "SACE_YBY"}) == {"YAL001C": {"AAA": 1.0, "SACE_YBY": 2.0}}
    rna.write_text(rna.read_text() + "YAL001C,5,6,7\n")
    with pytest.raises(ValueError, match="Repeated gene"):
        source.rna_covariates(rna, {"YAL001C"}, {"AAA"})


def test_archive_stream_filters_runs_and_verifies_integrity(tmp_path):
    runs, records, _ = fixture()
    data = tmp_path / source.MEMBER
    with data.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row()), delimiter="\t")
        writer.writeheader()
        for name, r in records + [("a", row()), ("b", row())]:
            writer.writerow(dict(r, **{"File.Name": "D:\\data\\" + name + ".rawIdx.wiff.dia"}))
    archive = tmp_path / "report.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(data, source.MEMBER)
    counts = Counter()
    selected = list(source.stream_report(archive, runs, qc=False, selected={"AAA"}, counters=counts))
    assert len(selected) == 1 and selected[0][0] == "a"
    assert counts["source_rows"] == len(records) + 2
    assert len(list(source.stream_report(archive, runs, qc=True, selected=set(), counters=Counter()))) == len(records)
    archive.write_bytes(b"corrupt archive")
    with pytest.raises(ValueError, match="Empty or invalid"):
        list(source.stream_report(archive, runs, qc=True, selected=set(), counters=Counter()))


def test_precursor_identity_conflicts_cannot_enter_panel():
    runs, records, aliases = fixture()
    aliases["G2"] = "YAL002W"
    records[0] = (records[0][0], dict(records[0][1], Genes="G2"))
    panel = source.qc_panel(records, runs, aliases, Counter())
    assert hashlib.sha256(b"P1").hexdigest() not in panel


def test_accession_mapping_requires_one_species_and_one_author_locus(tmp_path):
    path = tmp_path / "accessions.tsv"
    path.write_text(
        "Entry\tGene Names (ordered locus)\tGene Names (primary)\tOrganism (ID)\n"
        "P09938\tYJL026W\tRNR2\t559292\n"
        "P00001\tYJL026W YAL001C\t\t559292\n"
    )
    aliases = source.with_accessions(path, {"YJL026W": "YJL026W", "YAL001C": "YAL001C"})
    assert aliases["P09938"] == "YJL026W"
    assert "P00001" not in aliases
    actual = dict(row(gene=""), **{"Protein.Ids": "P09938"})
    assert source.qualified(actual, aliases)[1] == "YJL026W"
    assert source.qualified(dict(actual, Genes="YAL001C"), aliases) is None
    assert source.qualified(dict(actual, **{"Protein.Ids": "P09938;P00001"}), aliases) is None
    path.write_text(path.read_text().replace("559292", "9606"))
    with pytest.raises(ValueError, match="another organism"):
        source.with_accessions(path, aliases)
