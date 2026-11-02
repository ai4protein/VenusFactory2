"""Recompute published PET measurements and audit checkpoint/paper concordance.

This uses the authors' measurements. It neither generates wet-lab data nor
interprets successful notebook execution as reproduction of published rankings.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import io
import json
import re
import statistics
import zipfile
from pathlib import Path

import openpyxl
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_triplicates(sheet, *, metadata, replicates, mean_column, sd_column):
    values, current = [], {}
    for number, row in enumerate(sheet.iter_rows(min_row=5, values_only=True), 5):
        if all(row[index - 1] is None for index in replicates):
            continue
        for name, index in metadata.items():
            if row[index - 1] is not None:
                current[name] = row[index - 1]
        if set(current) != set(metadata):
            raise ValueError("missing condition metadata")
        measurements = [row[index - 1] for index in replicates]
        if any(type(value) not in (float, int) for value in measurements):
            raise ValueError("missing or nonnumeric replicate; no imputation allowed")
        mean, population_sd, sample_sd = (
            statistics.mean(measurements),
            statistics.pstdev(measurements),
            statistics.stdev(measurements),
        )
        stored_mean, stored_sd = row[mean_column - 1], row[sd_column - 1]
        if type(stored_mean) not in (float, int) or type(stored_sd) not in (float, int):
            raise ValueError("missing published summary values")
        values.append(
            {
                **current,
                "sheet": sheet.title,
                "source_row": number,
                "replicates": measurements,
                "mean": mean,
                "population_sd": population_sd,
                "sample_sd": sample_sd,
                "stored_mean": stored_mean,
                "stored_sd": stored_sd,
                "mean_abs_error": abs(mean - stored_mean),
                "population_sd_abs_error": abs(population_sd - stored_sd),
                "sample_sd_abs_error": abs(sample_sd - stored_sd),
            }
        )
    return values


def parse_rankings(text):
    names = ["BhrPETase_Uniclust30_BFD", "BhrPETase_UniProt", "LCCICCG_Uniclust30_BFD", "LCCICCG_UniProt"]
    tables = {name: [] for name in names}
    pattern = re.compile(
        r"^(10|[1-9])\s+([A-Z]\d+)\s+([0-9.]+)\s+([A-Z]\d+)\s+([0-9.]+)\s+([A-Z]\d+)\s+([0-9.]+)\s+([A-Z]\d+)\s+([0-9.]+)\s*$"
    )
    for line in text.splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            groups = match.groups()
            for i, name in enumerate(names):
                tables[name].append(
                    {"rank": int(groups[0]), "residue": groups[1 + i * 2], "score": float(groups[2 + i * 2])}
                )
    if any([row["rank"] for row in rows] != list(range(1, 11)) for rows in tables.values()):
        raise ValueError("Supplementary Table 1 must contain all ten published ranks")
    return tables


def main(args):
    manifest = json.loads((args.assets / "source_manifest.json").read_text())
    if manifest.get("status") != "completed" or manifest.get("doi") != "10.1038/s41467-024-45662-9":
        raise ValueError("verified paper source manifest required")
    for entry in manifest["files"]:
        if checksum(args.assets / entry["local_file"]) != entry["sha256"]:
            raise ValueError("paper source asset changed")
    args.output.mkdir(parents=True, exist_ok=False)
    archive = args.assets / "41467_2024_45662_MOESM4_ESM.zip"
    with zipfile.ZipFile(archive) as z:
        members = [{"name": info.filename, "bytes": info.file_size} for info in z.infolist() if not info.is_dir()]
        w2 = openpyxl.load_workbook(io.BytesIO(z.read("source data/Figure 2.xlsx")), data_only=True)
        w4 = openpyxl.load_workbook(io.BytesIO(z.read("source data/Figure 4.xlsx")), data_only=True)
        if (
            w2["Figure 2b"].cell(2, 5).value != "Released TPA + MHET (mM) in 3 h"
            or w4["Figure 4A"].cell(2, 6).value != "Depolymerization (%)"
        ):
            raise ValueError("source measurement header changed")
        fig2 = extract_triplicates(
            w2["Figure 2b"],
            metadata={"conditions": 2, "temperature_C": 3, "enzyme": 4},
            replicates=(5, 6, 7),
            mean_column=8,
            sd_column=9,
        )
        fig4 = extract_triplicates(
            w4["Figure 4A"],
            metadata={"conditions": 2, "temperature_C": 3, "enzyme": 4, "time_h": 5},
            replicates=(6, 7, 8),
            mean_column=9,
            sd_column=10,
        )
        scaleup = [
            {"source_row": i, "time_h": row[2], "conversion_percent_NaOH": row[3]}
            for i, row in enumerate(w4["Figure 4C"].values, 1)
            if i >= 5 and type(row[2]) in (float, int) and type(row[3]) in (float, int)
        ]
    reader = PdfReader(args.assets / "41467_2024_45662_MOESM1_ESM.pdf")
    candidates = [
        (i + 1, page.extract_text())
        for i, page in enumerate(reader.pages)
        if "Supplementary Table 1." in page.extract_text()
    ]
    if len(candidates) != 1:
        raise ValueError("ambiguous published ranking table")
    page_number, text = candidates[0]
    rankings = parse_rankings(text)
    replay = json.loads(args.inference.read_text())
    concordance = []
    for run in replay["comparisons"]:
        observed = [row["wild_type"] + str(row["position"]) for row in run["top10"]]
        for name in ["BhrPETase_Uniclust30_BFD", "BhrPETase_UniProt"]:
            expected = [row["residue"] for row in rankings[name]]
            concordance.append(
                {
                    "checkpoint": run["checkpoint"],
                    "indexing": run["indexing"],
                    "published_column": name,
                    "top10_overlap": len(set(observed) & set(expected)),
                    "same_order": observed == expected,
                    "shared_residues": sorted(set(observed) & set(expected)),
                }
            )
    results = fig2 + fig4
    outcome = {
        "status": "completed",
        "origin": "reanalysis of author-reported experimental measurements; no new wet-lab observations",
        "paper_doi": manifest["doi"],
        "source_archive_sha256": checksum(archive),
        "inference_report_sha256": checksum(args.inference),
        "figure2b": fig2,
        "figure4a": fig4,
        "figure4c": scaleup,
        "replicate_groups": len(results),
        "raw_replicate_values": sum(len(row["replicates"]) for row in results),
        "max_mean_abs_error": max(row["mean_abs_error"] for row in results),
        "max_population_sd_abs_error": max(row["population_sd_abs_error"] for row in results),
        "spread_convention": "source workbooks use population SD (ddof=0); sample SD (ddof=1) is also recorded, not silently substituted",
        "published_ranking_table": {"source_pdf_page": page_number, "columns": rankings},
        "checkpoint_to_published_rankings": concordance,
        "published_exact_top10_reproduced": any(row["same_order"] for row in concordance),
        "checkpoint_training_dataset_mapping": "not unambiguously documented; all BhrPETase columns compared without post-hoc checkpoint selection",
        "scoring_difference": "paper average divides 19 non-wild-type differences; notebook averages all 20 including zero wild-type difference; constant scaling alone cannot change rankings",
        "source_data_members": members,
        "training_sequence_file_in_source_data_archive": any(
            Path(row["name"]).suffix.lower() in {".fasta", ".fa", ".a3m", ".aln"} for row in members
        ),
        "missing_for_complete_computational_reproduction": [
            "original 10847-sequence and 15051-sequence training inputs",
            "exact training environment and mapping from released weights to paper columns",
            "GRAPE intermediate structures and FoldX/Rosetta/ABACUS/DDD inputs",
            "docking and MD starting systems with original run parameters",
        ],
        "full_paper_reproduced": False,
    }
    (args.output / "report.json").write_text(json.dumps(outcome, indent=2) + "\n")
    (args.output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    print(
        json.dumps(
            {
                key: outcome[key]
                for key in [
                    "status",
                    "replicate_groups",
                    "raw_replicate_values",
                    "max_mean_abs_error",
                    "max_population_sd_abs_error",
                    "published_exact_top10_reproduced",
                    "training_sequence_file_in_source_data_archive",
                ]
            }
        ),
        flush=True,
    )
    print(json.dumps(concordance), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "research_runs/rsi/case_assets/turbopetase_pmc_v1")
    parser.add_argument(
        "--inference",
        type=Path,
        default=ROOT / "research_runs/rsi/20261011_turbopetase_author_inference_v1/report.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
