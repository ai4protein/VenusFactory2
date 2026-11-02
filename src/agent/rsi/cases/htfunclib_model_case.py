"""All released GFP colour-classification rows for source-code model development.

The representation is the author's aligned 22-residue design-position string,
not an invented full protein sequence. All three published row partitions are
preserved. Only training rows enter development; acceptance and future data are
separate host-owned inputs.
"""

from __future__ import annotations

from agent.rsi import generated_classifier_large as classifier
from agent.rsi import htfunclib_case as source_case
from agent.rsi.engine.protein_model import digest


def prepare_htfunclib_model_case(source):
    protocol, _, _, identities = source_case.admit(source)
    labels = ["U", "C", "G", "G/C"]
    scope = {
        "protein": "htFuncLib GFP fixed-scaffold library",
        "assay": "published nohbonds NGS/FACS colour assignments",
        "target": "author fluorescence class from aligned design-position residues",
        "unit": "class probability",
        "sequence_length": 22,
        "class_labels": ["author_U", "author_C", "author_G", "author_G/C"],
    }
    groups = {}
    for source_name, destination in [("train", "training"), ("valid", "validation"), ("test", "future")]:
        rows = [
            {
                "id": digest({"dataset": "htfunclib_GFP", "sequence": str(row.pseudo_prot)}),
                "sequence": str(row.pseudo_prot),
                "target": labels.index(str(row.assign)),
            }
            for row in identities[source_name].itertuples(index=False)
        ]
        classifier.validate_rows(rows, scope, labels=True)
        if {r["target"] for r in rows} != set(range(4)):
            raise ValueError("A GFP partition is missing a declared class")
        groups[destination] = rows
    seen = set()
    for rows in groups.values():
        sequences = {r["sequence"] for r in rows}
        if seen & sequences:
            raise ValueError("GFP development, acceptance and future sequence overlap")
        seen |= sequences
    source_record = {
        "paper_doi": protocol["paper_doi"],
        "source_repository": protocol["source_repository"],
        "source_commit": protocol["source_commit"],
        "source_files_sha256": protocol["source_files_sha256"],
        "source_license": protocol["source_license"],
        "rows": protocol["rows"],
        "author_split_admission": protocol["partitions"],
    }
    split = {
        "training": "all 116116 author X_train row identities",
        "acceptance": "all 29030 author X_valid row identities; host development selection only",
        "future": "all 36287 author X_test row identities; score after every allocation is terminal",
        "representation": "all 22 variable-position amino acids in the author pseudo_prot order; fixed scaffold",
        "comparison_scope": "sequence-only generated model; the 74 engineered author classifier features are not supplied",
        "historical_scope": "public retrospective study, already examined by investigators; not blind or prospective",
        "class_order": labels,
        "counts": {k: len(v) for k, v in groups.items()},
        "data_hashes": {k: digest(v) for k, v in groups.items()},
        "class_counts": {k: [sum(row["target"] == c for row in v) for c in range(4)] for k, v in groups.items()},
        "history_projection": "empty; no internal prior development history, private donor source or fitted state",
    }
    return scope, groups["training"], groups["validation"], groups["future"], source_record, split
