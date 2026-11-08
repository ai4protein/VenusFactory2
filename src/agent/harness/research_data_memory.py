"""Recover verified prepared-data metadata across Scientist iterations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from utils.scoped_files import read_bytes
from .research_development import evaluation_digest
from .research_dataset import model_interface, protected_data_access


def prepared_dataset_context(workspace, episodes, *, data_store=None):
    """Return metadata from a host receipt only while both files still match it."""
    receipts = sorted(Path(episodes).glob("cycle_*/tools/receipts.json"), reverse=True)[:24]
    for receipt_path in receipts:
        try:
            rows = json.loads(receipt_path.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(rows, list):
            continue
        for row in reversed(rows):
            if not isinstance(row, dict) or row.get("tool") != "prepare_research_dataset":
                continue
            result = row.get("result")
            if not isinstance(result, dict) or result.get("passed") is not True:
                continue
            try:
                evaluation = data_store.load_evaluation(result["evaluation_file"]) if data_store else read_bytes(workspace, result["evaluation_file"], max_bytes=20 * 1024**2)
                table = data_store.original_bytes(result["source_file"]) if data_store else read_bytes(workspace, result["source_file"], max_bytes=20 * 1024**2)
                if (hashlib.sha256(evaluation).hexdigest() != result["source_sha256"]
                        or hashlib.sha256(table).hexdigest() != result["table_sha256"]):
                    continue
                data = json.loads(evaluation)
                if evaluation_digest(data) != result["evaluation_sha256"]:
                    continue
                preparation = data["preparation"]
                if (preparation["source_file"] != result["source_file"]
                        or preparation["source_sha256"] != result["table_sha256"]):
                    continue
                splits = {name: {"rows": len(values),
                                  "groups": len({entry["group"] for entry in values})}
                          for name, values in data["splits"].items()}
                columns = {name: preparation[name]
                           for name in ("target_column", "input_columns", "group_column", "id_column")}
                return {
                    "status": "verified", "source_file": result["source_file"],
                    "evaluation_file": result["evaluation_file"],
                    "evaluation_sha256": result["evaluation_sha256"],
                    "task": data["task"], "columns": columns, "splits": splits,
                    "model_interface": model_interface(data["task"], preparation["input_columns"]),
                    **(protected_data_access(data) if data_store else {}),
                    "input_type": "Original table strings; convert numerical features in fit/predict.",
                    "next_step": "Reuse this evaluation_file. Write the complete .py algorithm and train a baseline.",
                }
            except (OSError, ValueError, TypeError, KeyError):
                continue
    return None
