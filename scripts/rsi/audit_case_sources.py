"""Audit pinned public case assets without executing downloaded author code.

This is source preparation, not an agent evaluation or a completed reproduction.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

KEYS = {
    "analysis/ParPgb_fitness_all.csv": "Combo",
    "data/ParPgb/fitness_round1_training.csv": "Combo",
    "data/landscape.csv": "sequence",
    "designs_counts.csv": "seq_id",
    "designs_scores.csv": "seq_id",
}


def contained_file(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or path == root:
        raise ValueError("asset path escapes its source directory")
    return path


def verify_asset(root, asset):
    path = contained_file(root, asset["path"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != asset["sha256"] or path.stat().st_size != asset["bytes"]:
        raise ValueError("asset checksum or size mismatch: " + asset["path"])
    return path


def summarize_csv(path, key=None):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("missing or duplicate CSV headers")
        if key and key not in columns:
            raise ValueError("missing identifier column")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError("CSV row length differs from header")
    missing = {col: sum(row[col].strip().lower() in {"", "na", "nan", "null"} for row in rows)
               for col in columns}
    nonfinite = {}
    for col in columns:
        count = 0
        for row in rows:
            try:
                value = float(row[col])
            except ValueError:
                continue
            count += not math.isfinite(value)
        if count:
            nonfinite[col] = count
    result = {"rows": len(rows), "columns": columns, "missing_by_column": missing,
              "nonfinite_numeric_by_column": nonfinite}
    if key:
        ids = [row[key] for row in rows]
        result.update(identifier=key, unique_identifiers=len(set(ids)),
                      duplicate_identifier_rows=len(ids) - len(set(ids)),
                      missing_identifiers=sum(not x.strip() for x in ids))
    return result


def alde_trajectory(path):
    """Replay the author notebook's yield summary, not its model/acquisition loop."""
    groups = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            first, second = float(row["yield1"]), float(row["yield2"])
            if not math.isfinite(first) or not math.isfinite(second):
                raise ValueError("nonfinite yield")
            groups[row["round"]].append((first, second))
    if not groups:
        raise ValueError("empty ALDE trajectory")
    result = {}
    for name, rows in groups.items():
        best = max(rows, key=lambda pair: pair[0] - pair[1])
        ratios = [a / b for a, b in rows if b != 0]
        result[name] = {
            "observed_rows": len(rows),
            "mean_total_yield_percent": sum(a + b for a, b in rows) / len(rows),
            "mean_selectivity_defined_rows": sum(ratios) / len(ratios) if ratios else None,
            "undefined_selectivity_rows": len(rows) - len(ratios),
            "best_by_yield_difference_total_yield_percent": sum(best),
            "best_by_yield_difference_selectivity": best[0] / best[1] if best[1] else None,
        }
    return {"scope": "published processed measurements to trajectory summaries only",
            "agent_run": False, "full_author_baseline_completed": False,
            "wet_lab_reproduced": False, "rounds": result}


def audit_catalog(catalog, cache):
    reports = []
    for case in catalog["cases"]:
        item = {"id": case["id"], "repository": case["source"]["repository"],
                "commit": case["source"]["commit"], "assets": [], "errors": [],
                "author_baseline_completed": False, "agent_end_to_end_completed": False}
        try:
            folder = contained_file(cache, case["cache_subdir"])
            for asset in case["source"]["files"]:
                path = verify_asset(folder, asset)
                record = {"path": asset["path"], "sha256": asset["sha256"], "bytes": asset["bytes"]}
                if path.suffix == ".csv":
                    record["table"] = summarize_csv(path, KEYS.get(asset["path"]))
                item["assets"].append(record)
                if case["id"] == "alde_parpgb" and asset["path"] == "analysis/ParPgb_fitness_all.csv":
                    item["author_formula_replay"] = alde_trajectory(path)
        except (OSError, ValueError, KeyError, csv.Error) as error:
            item["errors"].append(str(error))
        item["source_integrity_passed"] = bool(item["assets"]) and not item["errors"]
        reports.append(item)
    return {"schema_version": 1, "stage": "source_audit_and_partial_formula_replay",
            "all_source_integrity_passed": bool(reports) and all(r["source_integrity_passed"] for r in reports),
            "cases": reports, "agent_evaluation_started": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text())
    report = audit_catalog(catalog, args.cache)
    report["catalog_sha256"] = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cases": len(report["cases"]), "source_integrity_passed": report["all_source_integrity_passed"],
                      "agent_evaluation_started": False, "report": str(args.output)}))
    return 0 if report["all_source_integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
