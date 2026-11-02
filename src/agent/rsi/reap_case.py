"""Read-only admission of publisher REAP sequences and replicate measurements.

No author code is executed. A campaign row needs a unique agreement between
its round-specific full sequence, plotting mean, and three raw replicates.
Ambiguous records and independent comparisons remain explicit and excluded.
"""
from __future__ import annotations

import hashlib
import io
import math
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

ARCHIVES = {
    "41467_2026_76264_MOESM7_ESM.zip": "4c8a9d3d2df11d5df946d2946a7bdeb36219841833882dffbe909b58cec687d0",
    "41467_2026_76264_MOESM10_ESM.zip": "59ab9b320d8baf8ab43f531e33e6a43df5fa392fb792b235c2581837eb52d3dc",
}
P450 = "P450_BM3_FL62"
SRTA = "SaSrtA"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
TOKEN = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])")


def mutation_key(name):
    tokens = str(name).strip().split("-")
    if not all(TOKEN.fullmatch(t) for t in tokens):
        raise ValueError("Invalid mutation label: " + str(name))
    positions = [int(t[1:-1]) for t in tokens]
    if len(set(positions)) != len(positions) or any(t[0] == t[-1] for t in tokens):
        raise ValueError("Duplicate position or identity substitution: " + str(name))
    return "-".join(sorted(tokens, key=lambda t: int(t[1:-1])))


def apply_mutations(reference, name, index_shift):
    result = list(reference)
    for token in mutation_key(name).split("-"):
        index = int(token[1:-1]) + index_shift
        if not 0 <= index < len(reference) or reference[index] != token[0]:
            raise ValueError("Reference residue mismatch: " + token)
        result[index] = token[-1]
    return "".join(result)


def close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def summarize_replicates(values, reported_mean, reported_sd):
    if len(values) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("Three finite measured replicates required")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    if not close(mean, reported_mean) or not close(sd, reported_sd):
        raise ValueError("Replicate summary differs from publisher mean or sample standard deviation")
    return mean, sd


def join_campaign(sequences, plots, assays):
    """Never resolve an ambiguous replicate match by row order or nearest value."""
    by_sequence, by_assay = defaultdict(list), defaultdict(list)
    for row in sequences:
        if row["role"] == "campaign" and row["sequence_consistent"]:
            by_sequence[(row["family"], row["round"], row["key"])].append(row)
    for row in assays:
        by_assay[(row["family"], row["key"])].append(row)
    admitted, exclusions, controls, used = [], [], [], set()
    for plot in plots:
        group = (plot["family"], plot["round"], plot["key"])
        seqs = by_sequence[group]
        if not seqs and plot["control"]:
            controls.append(plot)
            continue
        matches = [a for a in by_assay[(plot["family"], plot["key"])] if close(a["mean"], plot["mean"])]
        anchor = plot.get("replicate_anchor")
        if anchor:
            matches = [a for a in matches if all(close(x, y) for x, y in zip(a["replicates"], anchor["replicates"], strict=True))]
        if len(seqs) != 1 or len(matches) != 1 or matches[0]["id"] in used:
            exclusions.append({"plot": plot, "sequence_matches": len(seqs), "assay_ids": [a["id"] for a in matches], "reason": "nonunique_or_missing_sequence_replicate_agreement"})
            continue
        seq, assay = seqs[0], matches[0]
        used.add(assay["id"])
        admitted.append({
            "id": assay["id"], "family": plot["family"], "round": plot["round"],
            "name": plot["name"], "key": plot["key"], "sequence": seq["sequence"],
            "sequence_sha256": seq["sequence_sha256"], "mutation_order": len(plot["key"].split("-")),
            "replicates": assay["replicates"], "target": assay["mean"], "sample_sd": assay["sample_sd"],
            "sequence_source": seq["source"], "plot_source": plot["source"], "assay_source": assay["source"],
            "independent_replicate_anchor": anchor,
            "assay_context": ("purified_initial_selection" if plot["family"] == P450 else "initial_selection_assay") if plot["round"] == 0 else "normalized_lysate_automated_round",
        })
    return admitted, exclusions, controls, [a for a in assays if a["id"] not in used]


class PublisherData:
    def __init__(self, assets):
        self.assets = Path(assets)
        self.members = {}
        for name, expected in ARCHIVES.items():
            if hashlib.sha256((self.assets / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Publisher archive hash mismatch: " + name)

    def workbook(self, archive, contains):
        with zipfile.ZipFile(self.assets / archive) as z:
            members = [n for n in z.namelist() if contains in n and n.endswith(".xlsx")]
            if len(members) != 1:
                raise ValueError("Unique publisher workbook required: " + contains)
            name = members[0]
            payload = z.read(name)
        sha = hashlib.sha256(payload).hexdigest()
        self.members[name] = {"archive": archive, "archive_sha256": ARCHIVES[archive], "sha256": sha}
        return openpyxl.load_workbook(io.BytesIO(payload), read_only=True, data_only=True), {"archive": archive, "member": name, "sha256": sha}


def location(source, sheet, row, column):
    return {**source, "sheet": sheet, "row": row, "column": column}


def ingest(assets):
    source = PublisherData(assets)
    sequence_archive, assay_archive = ARCHIVES
    refs, ref_sources, sequences, plots, assays, issues = {}, {}, [], [], [], []
    wb, origin = source.workbook(sequence_archive, "_1_")
    rows = list(wb["Amino acid sequences"].values)
    reference_row = next(i for i, row in enumerate(rows, 1) if row[0] == "FL#62")
    raw_reference = rows[reference_row - 1][2]
    if not raw_reference.startswith("M") or not raw_reference.endswith("*"):
        raise ValueError("Unexpected P450 reference construct")
    refs[P450] = raw_reference[1:-1]
    ref_sources[P450] = {"source": location(origin, "Amino acid sequences", reference_row, 3), "normalization": "remove annotated initial M and terminal stop to match all published variant constructs", "index_shift": -1}
    wb.close()
    for family, part, shift in [(P450, "_3_", -1), (SRTA, "_4_", -59)]:
        wb, origin = source.workbook(sequence_archive, part)
        if family == SRTA:
            rows = list(wb["wtSaSrtA"].values)
            refs[family] = rows[2][1]
            ref_sources[family] = {"source": location(origin, "wtSaSrtA", 3, 2), "normalization": "published truncated construct with initiating M retained", "index_shift": shift}
        for sheet in wb:
            if sheet.title == "wtSaSrtA":
                continue
            round_ = 0 if sheet.title == "Zero-shot" else (int(sheet.title[-1]) if re.fullmatch(r"Round[1-5]", sheet.title) else None)
            role = "campaign" if round_ is not None else {"Additional saturation mutations": "independent_saturation", "Quintuple variant": "independent_quintuple"}[sheet.title]
            rows = list(sheet.values)
            headers = [(i, c) for i, row in enumerate(rows[:4]) for c in range(len(row)-1) if row[c:c+2] == ("Mutant", "Sequence")]
            if not headers:
                raise ValueError("Sequence header missing")
            for header, col in headers:
                for ri in range(header + 1, len(rows)):
                    name, sequence = rows[ri][col:col+2]
                    if name is None and sequence is None:
                        continue
                    key = mutation_key(name)
                    if not isinstance(sequence, str) or not sequence or set(sequence) - AA:
                        raise ValueError("Malformed full protein sequence")
                    loc = location(origin, sheet.title, ri+1, col+1)
                    try:
                        consistent = apply_mutations(refs[family], key, shift) == sequence
                    except ValueError:
                        consistent = False
                    if not consistent:
                        issues.append({"reason": "sequence_mutation_disagreement", "source": loc, "name": name})
                    sequences.append({"family": family, "role": role, "round": round_, "name": name, "key": key, "sequence": sequence, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(), "sequence_consistent": consistent, "source": loc})
        wb.close()
    for family, part, sheet_name in [(P450, "Figure 3.xlsx", "Fig.3d"), (SRTA, "Source Data_Supplementary Information.xlsx", "Supplementary Fig.17b")]:
        wb, origin = source.workbook(assay_archive, part)
        rows = list(wb[sheet_name].values)
        if rows[2][:4] != ("Mutant", "Replicate1", "Replicate2", "Replicate3"):
            raise ValueError("Unexpected raw replicate header")
        for ri, row in enumerate(rows[3:], 4):
            if row[0] is None:
                continue
            key = mutation_key(row[0])
            values = list(row[1:4])
            mean, sd = summarize_replicates(values, row[4], row[5])
            assays.append({"id": f"{family}:{sheet_name}:A{ri}", "family": family, "name": row[0], "key": key, "replicates": values, "mean": mean, "sample_sd": sd, "source": location(origin, sheet_name, ri, 1)})
        wb.close()
    for family, number in [(P450, 3), (SRTA, 6)]:
        wb, origin = source.workbook(assay_archive, f"Figure {number}.xlsx")
        sheet_name = f"Fig.{number}a"
        rows = list(wb[sheet_name].values)
        for round_, col in enumerate([0, 3, 6, 9, 13, 17]):
            for ri, row in enumerate(rows[6:], 7):
                name, value = row[col:col+2]
                if name is None or (name, value) == ("Mutant", "Fold change"):
                    continue
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Nonfinite plot target")
                try:
                    key = mutation_key(name)
                    control = name == "S81L" and round_ > 0 and family == P450
                except ValueError:
                    if name not in {"28a", "pET28a", "FL#62", "wtSaSrtA", "eSrtA"}:
                        raise
                    key, control = name, True
                plots.append({"family": family, "round": round_, "name": name, "key": key, "mean": value, "control": control, "source": location(origin, sheet_name, ri, col+1)})
        wb.close()
    # A separate purified-assay table disambiguates coincident means.
    wb, origin = source.workbook(sequence_archive, "_2_")
    rows = list(wb["Zero-shot prediction"].values)
    if rows[1][:5] != ("Mutant", "Prediction score", "Replicate1", "Replicate2", "Replicate3"):
        raise ValueError("Unexpected purified assay header")
    anchors = {}
    expected_anchors = {v["key"] for v in sequences if v["family"] == P450 and v["round"] == 0}
    for ri, row in enumerate(rows[2:], 3):
        if row[0] not in expected_anchors:
            continue
        key = mutation_key(row[0])
        values = list(row[2:5])
        mean, _ = summarize_replicates(values, row[5], row[6])
        if key in anchors:
            raise ValueError("Duplicate purified assay anchor")
        anchors[key] = {"replicates": values, "mean": mean, "source": location(origin, "Zero-shot prediction", ri, 1)}
    wb.close()
    if set(anchors) != expected_anchors:
        raise ValueError("Missing purified assay anchor for a published Round 0 variant")
    for plot in plots:
        if plot["family"] == P450 and plot["round"] == 0 and not plot["control"]:
            anchor = anchors.get(plot["key"])
            if not anchor or not close(anchor["mean"], plot["mean"]):
                raise ValueError("Purified assay anchor differs from Round 0 plot")
            plot["replicate_anchor"] = anchor
    admitted, exclusions, controls, remaining = join_campaign(sequences, plots, assays)
    summary = {
        "status": "partial_admission" if issues or exclusions else "admitted",
        "publisher": "https://www.nature.com/articles/s41467-026-76264-2",
        "archives": ARCHIVES, "workbooks": source.members, "reference_constructs": ref_sources,
        "sequence_counts": dict(Counter(f'{s["family"]}:{s["role"]}' for s in sequences)),
        "raw_assays": len(assays), "admitted_campaign": len(admitted),
        "admitted_by_round": dict(Counter(f'{s["family"]}:R{s["round"]}' for s in admitted)),
        "excluded_campaign_plots": len(exclusions), "plot_controls": len(controls), "unassigned_assays": len(remaining),
        "sequence_issues": len(issues), "author_replication_complete": False,
        "policy": "Only admitted campaign records enter subsequent experimental-round training. Independent saturation/quintuple variants, controls and unresolved assays are never automatically added. Keep assay context; do not pool replicate triplets or guess row-order mappings. Exact author replication remains incomplete while any campaign record is unresolved.",
    }
    return {"summary": summary, "campaign": admitted, "exclusions": exclusions, "controls": controls, "unassigned_assays": remaining, "sequences": sequences, "issues": issues}
