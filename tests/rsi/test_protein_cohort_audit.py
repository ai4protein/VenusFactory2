import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.rsi import protein_cohort_audit as a


def record(ident, cohort, sequence, labels=None):
    return {"id": ident, "cohort": cohort, "sequence": sequence, "labels": labels or ["one"]}


def fixture_records():
    return [record("a", "train", "ACDEFGHIKL"), record("b", "test", "ACDEFGHIKL")]


def hsp(q="ACDEFGHIKL", s="ACDEFGHIKL", **changes):
    values = {
        "qid": "q00000001",
        "sid": "t00000001",
        "length": len(q),
        "qlen": 10,
        "slen": 10,
        "qs": 1,
        "qe": 10,
        "ss": 1,
        "se": 10,
        "ev": "1e-10",
        "bits": 40,
        "q": q,
        "s": s,
    }
    values.update(changes)
    return "\t".join(str(v) for v in values.values()) + "\n"


def test_fixed_cohorts_and_labels():
    rows = fixture_records() + [
        record("c", "test", "CDEFGHIKLM", ["new", "one"]),
        record("d", "test", "DEFGHIKLMN", ["new"]),
    ]
    rows += [record("e", "test", "ACDEFGHIKL", ["conflict"])]
    result = a.cohort_summary(rows, "train")
    test = result["cohorts"]["test"]
    assert (test["all_labels_seen_rows"], test["partly_seen_rows"], test["no_labels_seen_rows"]) == (1, 1, 2)
    assert test["unseen_label_memberships"] == 3
    assert test["same_sequence_label_conflict_groups"] == 1
    assert result["exact_overlap"]["train|test"]["right_rows"] == 2


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p["records"].append(p["records"][0]),
        lambda p: p["records"][0].update(sequence="acde"),
        lambda p: p["records"][0].update(sequence="AC-D"),
        lambda p: p["records"][0].update(labels=["one", "one"]),
        lambda p: p.update(counts={"train": 2, "test": 1}),
    ],
)
def test_malformed_plan_is_rejected(mutation):
    plan = {"records": fixture_records()}
    mutation(plan)
    with pytest.raises(ValueError):
        a.select_records(plan, "train", ["test"])


def test_source_spans_and_gap_denominator():
    rows = [record("a", "train", "ACDEFGHIKL"), record("b", "test", "ACDFGHIKLM")]
    mapping = a.token_map(rows, "train")
    hit = a.parse_hsp(hsp("ACD-FGHIKL", "ACDEFGHIKL", qe=9), mapping)
    assert hit["identical_canonical_columns"] == 9
    assert hit["identity_percent"] == 90
    assert hit["query_coverage_percent"] == 90


@pytest.mark.parametrize(
    "change",
    [
        {"qid": "q99999999"},
        {"sid": "q00000001"},
        {"qlen": 11},
        {"slen": 11},
        {"qs": 0},
        {"qe": 9},
        {"se": 11},
        {"length": 9},
        {"q": "ACDEFGHIKM"},
        {"s": "ACDEFGHIKM"},
        {"ev": "nan"},
        {"ev": 11},
        {"bits": "inf"},
        {"bits": -1},
    ],
)
def test_tampered_alignment_rejected(change):
    with pytest.raises(ValueError):
        a.parse_hsp(hsp(**change), a.token_map(fixture_records(), "train"))


def test_ambiguities_do_not_inflate_identity():
    rows = [record("a", "train", "ACDEFGHIOU"), record("b", "test", "ACDEFGHIXX")]
    hit = a.parse_hsp(hsp("ACDEFGHIXX", "ACDEFGHIXX"), a.token_map(rows, "train"))
    assert hit["identity_percent"] == 80
    assert a.cohort_summary(rows, "train")["exact_overlap"]["train|test"]["shared_unique_sequences"] == 0


def test_all_hsps_and_no_hit_rows_retained(tmp_path):
    rows = fixture_records() + [record("c", "test", "M" * 10)]
    mapping = a.token_map(rows, "train")
    path = tmp_path / "hits.tsv"
    # The best bit score does not cover enough of either sequence.
    path.write_text(hsp("ACDE", "ACDE", qe=4, se=4, bits=100) + hsp())
    report = a.summarize_alignments(path, mapping, "train")
    assert report["rows"][0]["best_bitscore_hsp"]["query_coverage_percent"] == 40
    assert report["rows"][0]["best_covering_hsp"]["query_coverage_percent"] == 100
    assert report["cohorts"]["test"]["matching_rows_at_identity_percent"]["90"] == 1
    assert report["cohorts"]["test"]["no_reported_hsp_rows"] == 1
    assert report["homology_independence_established"] is False


def test_empty_hits_and_both_coverage_required(tmp_path):
    rows = fixture_records()
    rows[0]["sequence"] *= 2
    mapping = a.token_map(rows, "train")
    path = tmp_path / "hits.tsv"
    path.write_text(hsp(slen=20))
    report = a.summarize_alignments(path, mapping, "train")
    assert report["cohorts"]["test"]["no_covering_hsp_rows"] == 1
    path.write_text("")
    assert a.summarize_alignments(path, mapping, "train")["cohorts"]["test"]["no_reported_hsp_rows"] == 1


def test_names_cannot_become_fasta_instructions(tmp_path):
    rows = fixture_records()
    rows[0]["id"] = "record with spaces | punctuation"
    a.write_fastas(tmp_path, a.token_map(rows, "train"))
    assert (tmp_path / "train.fasta").read_text().startswith(">t00000001\n")


def test_full_cli_with_real_blast(tmp_path):
    blast = Path(os.environ.get("RSI_TEST_BLAST_BIN", "/nonexistent"))
    if not (blast / "blastp").is_file():
        pytest.skip("Set RSI_TEST_BLAST_BIN for the real local BLAST integration test")
    import random

    rng = random.Random(17)
    sequence = "".join(rng.choice(sorted(a.CANONICAL)) for _ in range(200))
    plan = {"records": [record("t", "train", sequence), record("q", "test", sequence)]}
    research = tmp_path / "research_runs/rsi"
    research.mkdir(parents=True)
    source = research / "plan.json"
    source.write_text(json.dumps(plan))
    runner = Path(__file__).resolve().parents[2] / "scripts/rsi/run_protein_cohort_audit.py"
    output = research / "audit"
    cmd = [
        sys.executable,
        str(runner),
        "--workspace-root",
        str(tmp_path),
        "--plan",
        str(source),
        "--plan-sha256",
        a.sha(source),
        "--output",
        str(output),
        "--blast-bin",
        str(blast),
        "--evaluation-cohorts",
        "test",
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=90)
    result = json.loads((output / "sequence_audit.json").read_text())
    assert result["cohorts"]["test"]["matching_rows_at_identity_percent"]["90"] == 1
    before = a.sha(output / "report.json")
    rerun = subprocess.run(cmd, capture_output=True, timeout=30)
    assert rerun.returncode != 0
    assert a.sha(output / "report.json") == before
    manifest = json.loads((output / "report.json").read_text())["artifact_sha256"]
    assert all(a.sha(output / name) == expected for name, expected in manifest.items())
