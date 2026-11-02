"""Describe fixed protein cohorts; never change a split or certify non-homology."""

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
ALPHABET = CANONICAL | frozenset("BJOUXZ")
FIELDS = "qseqid sseqid length qlen slen qstart qend sstart send evalue bitscore qseq sseq"
THRESHOLDS = (30, 50, 80, 90)
COVERAGE_PERCENT = 80
BOUNDARY = (
    "Descriptive audit of fixed cohorts, not a new split or a performance comparison. "
    "BLASTP is a heuristic local search. No hit, or no single HSP spanning both sequences, "
    "does not prove absence of homology. Thresholds are operational descriptions, not "
    "universal biological definitions. No evaluation rows are removed."
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def select_records(plan, train_cohort, evaluation_cohorts):
    """Read explicit cohorts without interpreting publication split names."""
    names = [train_cohort, *evaluation_cohorts]
    if len(names) < 2 or len(set(names)) != len(names) or any(not isinstance(x, str) or not x for x in names):
        raise ValueError("Distinct training and evaluation cohorts required")
    if not isinstance(plan.get("records"), list):
        raise ValueError("Plan records must be a list")
    selected, seen, counts = [], set(), Counter()
    for raw in plan["records"]:
        if not isinstance(raw, dict):
            raise ValueError("Malformed plan record")
        cohort = raw.get("cohort")
        if cohort not in names:
            continue
        ident, seq, labels = raw.get("id"), raw.get("sequence"), raw.get("labels")
        if not isinstance(ident, str) or not ident or any(ord(c) < 32 for c in ident):
            raise ValueError("Invalid record identifier")
        if (cohort, ident) in seen:
            raise ValueError("Repeated identifier within cohort")
        if not isinstance(seq, str) or not seq or not set(seq) <= ALPHABET:
            raise ValueError("Uppercase, ungapped protein sequence required")
        if not isinstance(labels, list) or not labels or any(not isinstance(x, str) or not x.strip() for x in labels):
            raise ValueError("Nonempty label list required")
        if len(set(labels)) != len(labels):
            raise ValueError("Repeated label in record")
        seen.add((cohort, ident))
        counts[cohort] += 1
        selected.append({"cohort": cohort, "id": ident, "sequence": seq, "labels": sorted(labels)})
    if set(counts) != set(names):
        raise ValueError("A selected cohort is empty or absent")
    if "counts" in plan and any(plan["counts"].get(k) != v for k, v in counts.items()):
        raise ValueError("Selected cohort count differs from plan")
    return selected


def normalize(sequence):
    # Explicitly mask ambiguity/rare residues for search; retain original strings for exact overlap.
    return "".join(c if c in CANONICAL else "X" for c in sequence)


def token_map(records, train_cohort):
    result, counters = {}, Counter()
    for row in records:
        prefix = "t" if row["cohort"] == train_cohort else "q"
        counters[prefix] += 1
        result[f"{prefix}{counters[prefix]:08d}"] = row
    return result


def write_fastas(root, mapping):
    root = Path(root)
    for prefix, name in [("t", "train.fasta"), ("q", "evaluation.fasta")]:
        with (root / name).open("x") as handle:
            for token, row in mapping.items():
                if token.startswith(prefix):
                    handle.write(f">{token}\n{normalize(row['sequence'])}\n")


def cohort_summary(records, train_cohort):
    groups = defaultdict(list)
    for row in records:
        groups[row["cohort"]].append(row)
    train = groups[train_cohort]
    train_labels = set().union(*(set(r["labels"]) for r in train))
    result = {"training_label_count": len(train_labels), "cohorts": {}, "exact_overlap": {}}
    for name, rows in groups.items():
        membership = Counter(label for r in rows for label in r["labels"])
        seqs = defaultdict(list)
        for r in rows:
            seqs[r["sequence"]].append(r)
        unseen = {k: v for k, v in sorted(membership.items()) if k not in train_labels}
        count_all = sum(set(r["labels"]) <= train_labels for r in rows)
        count_none = sum(not set(r["labels"]) & train_labels for r in rows)
        result["cohorts"][name] = {
            "rows": len(rows),
            "unique_labels": len(membership),
            "label_memberships": sum(membership.values()),
            "seen_label_memberships": sum(v for k, v in membership.items() if k in train_labels),
            "unseen_label_memberships": sum(unseen.values()),
            "unseen_labels": unseen,
            "all_labels_seen_rows": count_all,
            "no_labels_seen_rows": count_none,
            "partly_seen_rows": len(rows) - count_all - count_none,
            "unique_sequences": len(seqs),
            "duplicate_sequence_excess_rows": len(rows) - len(seqs),
            "duplicate_sequence_groups": sum(len(v) > 1 for v in seqs.values()),
            "same_sequence_label_conflict_groups": sum(len({tuple(r["labels"]) for r in v}) > 1 for v in seqs.values()),
            "masked_residues": sum(sum(c not in CANONICAL for c in r["sequence"]) for r in rows),
        }
    for a, b in itertools.combinations(groups, 2):
        aset, bset = {r["sequence"] for r in groups[a]}, {r["sequence"] for r in groups[b]}
        common = aset & bset
        result["exact_overlap"][a + "|" + b] = {
            "shared_identifiers": len({r["id"] for r in groups[a]} & {r["id"] for r in groups[b]}),
            "shared_unique_sequences": len(common),
            "left_rows": sum(r["sequence"] in common for r in groups[a]),
            "right_rows": sum(r["sequence"] in common for r in groups[b]),
        }
    return result


def parse_hsp(line, mapping):
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 13:
        raise ValueError("Expected all 13 BLAST fields")
    q, s = fields[:2]
    if q not in mapping or s not in mapping or not q.startswith("q") or not s.startswith("t"):
        raise ValueError("Unexpected alignment identifier")
    length, qlen, slen, qs, qe, ss, se = map(int, fields[2:9])
    ev, bits = map(float, fields[9:11])
    qa, sa = (x.upper() for x in fields[11:])
    query, subject = normalize(mapping[q]["sequence"]), normalize(mapping[s]["sequence"])
    if not math.isfinite(ev) or not math.isfinite(bits) or not 0 <= ev <= 10 or bits <= 0:
        raise ValueError("Invalid alignment score")
    if qlen != len(query) or slen != len(subject) or not 1 <= qs <= qe <= qlen or not 1 <= ss <= se <= slen:
        raise ValueError("Invalid sequence lengths or coordinates")
    if length <= 0 or len(qa) != length or len(sa) != length:
        raise ValueError("Alignment length differs")
    if qa.replace("-", "") != query[qs - 1 : qe] or sa.replace("-", "") != subject[ss - 1 : se]:
        raise ValueError("Alignment does not reproduce source sequence")
    if any(a == b == "-" for a, b in zip(qa, sa)):
        raise ValueError("Double-gap alignment column")
    # Count only exact canonical matches; X-X and ambiguities contribute zero.
    identical = sum(a == b and a in CANONICAL for a, b in zip(qa, sa))
    return {
        "query": q,
        "subject": s,
        "alignment_columns": length,
        "identical_canonical_columns": identical,
        "identity_percent": 100 * identical / length,
        "query_coverage_percent": 100 * (qe - qs + 1) / qlen,
        "subject_coverage_percent": 100 * (se - ss + 1) / slen,
        "evalue": ev,
        "bitscore": bits,
        "query_start": qs,
        "query_end": qe,
        "subject_start": ss,
        "subject_end": se,
        "query_length": qlen,
        "subject_length": slen,
    }


def covers(hit):
    return (
        100 * (hit["query_end"] - hit["query_start"] + 1) >= COVERAGE_PERCENT * hit["query_length"]
        and 100 * (hit["subject_end"] - hit["subject_start"] + 1) >= COVERAGE_PERCENT * hit["subject_length"]
    )


def summarize_alignments(path, mapping, train_cohort):
    train_labels = set().union(*(set(r["labels"]) for r in mapping.values() if r["cohort"] == train_cohort))
    rows = {}
    for token, row in mapping.items():
        if token.startswith("q"):
            rows[token] = {
                "token": token,
                "id": row["id"],
                "cohort": row["cohort"],
                "sequence_sha256": digest(row["sequence"]),
                "unseen_labels": sorted(set(row["labels"]) - train_labels),
                "hsp_count": 0,
                "best_bitscore_hsp": None,
                "best_covering_hsp": None,
                "threshold_matches": {str(t): False for t in THRESHOLDS},
            }
    count = 0
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            try:
                hit = parse_hsp(line, mapping)
            except (ValueError, KeyError) as exc:
                raise ValueError(f"Invalid HSP at line {number}: {exc}") from exc
            count += 1
            row = rows[hit["query"]]
            row["hsp_count"] += 1
            hit["subject_id"] = mapping[hit["subject"]]["id"]
            hit["subject_sequence_sha256"] = digest(mapping[hit["subject"]]["sequence"])
            old = row["best_bitscore_hsp"]
            if old is None or (hit["bitscore"], hit["identity_percent"]) > (old["bitscore"], old["identity_percent"]):
                row["best_bitscore_hsp"] = hit
            if covers(hit):
                old = row["best_covering_hsp"]
                if old is None or (hit["identity_percent"], hit["bitscore"]) > (
                    old["identity_percent"],
                    old["bitscore"],
                ):
                    row["best_covering_hsp"] = hit
                for t in THRESHOLDS:
                    if 100 * hit["identical_canonical_columns"] >= t * hit["alignment_columns"]:
                        row["threshold_matches"][str(t)] = True
    groups = defaultdict(list)
    for row in rows.values():
        groups[row["cohort"]].append(row)
    return {
        "validated_hsps": count,
        "rows": list(rows.values()),
        "cohorts": {
            name: {
                "rows": len(values),
                "no_reported_hsp_rows": sum(v["hsp_count"] == 0 for v in values),
                "no_covering_hsp_rows": sum(v["best_covering_hsp"] is None for v in values),
                "matching_rows_at_identity_percent": {
                    str(t): sum(v["threshold_matches"][str(t)] for v in values) for t in THRESHOLDS
                },
            }
            for name, values in groups.items()
        },
        "homology_independence_established": False,
        "boundary": BOUNDARY,
    }


def search_arguments(blastp, train_count):
    return [
        str(blastp),
        "-task",
        "blastp",
        "-query",
        "evaluation.fasta",
        "-db",
        "train_db",
        "-word_size",
        "2",
        "-evalue",
        "10",
        "-matrix",
        "BLOSUM62",
        "-gapopen",
        "11",
        "-gapextend",
        "1",
        "-comp_based_stats",
        "2",
        "-seg",
        "yes",
        "-soft_masking",
        "true",
        "-max_target_seqs",
        str(train_count),
        "-num_threads",
        "2",
        "-outfmt",
        "6 " + FIELDS,
        "-out",
        "alignments.tsv",
    ]
