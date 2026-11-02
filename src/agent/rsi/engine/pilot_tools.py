"""Small CPU-only MCP tool surface for engineering pilots, not the full VF benchmark.

This file is copied into a read-only capability bundle with the existing
calculate_physchem module. No hidden evaluator or test labels are bundled.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("VenusFactoryRSIPilot")
ROOT = Path(os.environ.get("VENUS_RSI_WORKSPACE", "/workspace")).resolve()
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def child(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        path = path.relative_to(ROOT)
    if ".." in path.parts:
        raise ValueError("path traversal")
    result = ROOT / path
    cursor = ROOT
    for part in path.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError("symlink outside artifact contract")
    result.resolve().relative_to(ROOT)
    return result


def rows(path: str) -> list[dict]:
    with child(path).open(newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, allow_nan=False, indent=2))
    os.replace(temp, path)


@mcp.tool()
def sequence_properties(fasta_file: str) -> dict:
    """Calculate an unambiguous single standard-amino-acid sequence using VF's existing ProtParam function."""
    from Bio import SeqIO
    from calculate_physchem import calculate_physchem_from_fasta

    path = child(fasta_file)
    records = list(SeqIO.parse(path, "fasta"))
    if len(records) != 1:
        return {"valid": False, "error": "exactly_one_sequence_required"}
    sequence = str(records[0].seq)
    invalid = sorted(set(sequence) - AA)
    if invalid:
        return {"valid": False, "error": "nonstandard_residues", "invalid_residues": invalid}
    if not sequence:
        return {"valid": False, "error": "empty_sequence"}
    return {"valid": True, "properties": calculate_physchem_from_fasta(str(path))}


@mcp.tool()
def audit_prediction_table(input_csv: str, prediction_csv: str) -> dict:
    """Check prediction sample IDs against input IDs, including duplicate, extra and missing IDs."""
    inputs, predictions = rows(input_csv), rows(prediction_csv)
    expected = [r["id"] for r in inputs]
    observed = [r["id"] for r in predictions]
    issues = {
        "missing_ids": sorted(set(expected) - set(observed)),
        "extra_ids": sorted(set(observed) - set(expected)),
        "duplicate_ids": sorted({i for i in observed if observed.count(i) > 1}),
        "duplicate_input_ids": sorted({i for i in expected if expected.count(i) > 1}),
    }
    return {"valid": not any(issues.values()), **issues}


@mcp.tool()
def convert_temperature(value: float, source_unit: str, target_unit: str) -> dict:
    """Convert between Celsius and Kelvin, rejecting values below absolute zero."""
    units = {"c": "C", "celsius": "C", "k": "K", "kelvin": "K"}
    source, target = units.get(source_unit.lower()), units.get(target_unit.lower())
    if not source or not target:
        raise ValueError("supported units are C and K")
    kelvin = float(value) + (273.15 if source == "C" else 0)
    if not np.isfinite(kelvin) or kelvin < 0:
        raise ValueError("invalid absolute temperature")
    return {"value": round(kelvin - (273.15 if target == "C" else 0), 6), "unit": target}


@mcp.tool()
def check_mutation(sequence: str, mutation: str) -> dict:
    """Validate a one-based substitution such as A1V against its wild-type sequence."""
    match = re.fullmatch(r"([ACDEFGHIKLMNPQRSTVWY])([1-9][0-9]*)([ACDEFGHIKLMNPQRSTVWY])", mutation)
    if not match or set(sequence) - AA:
        return {"valid": False, "error": "invalid_format_or_sequence"}
    old, pos, new = match.group(1), int(match.group(2)), match.group(3)
    if pos > len(sequence):
        return {"valid": False, "error": "position_out_of_range"}
    if sequence[pos - 1] != old:
        return {"valid": False, "error": "wildtype_mismatch", "observed": sequence[pos - 1]}
    return {"valid": True, "position": pos, "mutated_sequence": sequence[: pos - 1] + new + sequence[pos:]}


def model_path(model_id: str, registered: bool = False) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", model_id):
        raise ValueError("invalid model_id")
    return child(f"{'registry' if registered else 'staging'}/{model_id}.json")


@mcp.tool()
def fit_linear_fixture(dataset_csv: str, model_id: str) -> dict:
    """Fit a CPU linear regression to x1,x2,y CSV; synthetic engineering fixture only. Does NOT register the model."""
    data = rows(dataset_csv)
    if len(data) < 3:
        raise ValueError("at least three training rows required")
    x = np.array([[1.0, float(r["x1"]), float(r["x2"])] for r in data])
    y = np.array([float(r["y"]) for r in data])
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite data")
    if np.linalg.matrix_rank(x) < 3:
        raise ValueError("rank-deficient fixture")
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    path = model_path(model_id)
    if path.exists():
        raise ValueError("model_id already exists")
    card = {
        "schema": 1,
        "model_id": model_id,
        "status": "trained",
        "task": "synthetic_linear_fixture",
        "coefficients": coef.tolist(),
        "features": ["x1", "x2"],
        "unit": "arbitrary",
        "training_sha256": hashlib.sha256(child(dataset_csv).read_bytes()).hexdigest(),
    }
    write_json(path, card)
    return {"model_id": model_id, "status": "trained", "registered": False}


@mcp.tool()
def validate_linear_fixture(model_id: str, validation_csv: str) -> dict:
    """Validate a fixture on separate development rows, then register if RMSE <= 1e-6. Never uses final hidden labels."""
    source = model_path(model_id)
    card = json.loads(source.read_text())
    digest = hashlib.sha256(child(validation_csv).read_bytes()).hexdigest()
    if digest == card["training_sha256"]:
        raise ValueError("validation file equals training file")
    data = rows(validation_csv)
    if not data:
        raise ValueError("empty validation data")
    x = np.array([[1.0, float(r["x1"]), float(r["x2"])] for r in data])
    y = np.array([float(r["y"]) for r in data])
    error = float(np.sqrt(np.mean((x @ np.array(card["coefficients"]) - y) ** 2)))
    if not np.isfinite(error) or error > 1e-6:
        return {"validated": False, "registered": False, "rmse": error if np.isfinite(error) else None}
    card.update(status="validated", validation_sha256=digest, validation_rmse=error)
    target = model_path(model_id, registered=True)
    if target.exists():
        raise ValueError("registered model IDs are immutable")
    write_json(target, card)
    return {"validated": True, "registered": True, "model_id": model_id, "rmse": error}


@mcp.tool()
def predict_linear_fixture(model_id: str, input_csv: str, task: str = "synthetic_linear_fixture") -> dict:
    """Reuse a registered, validated fixture only for its declared task; preserve input sample IDs."""
    path = model_path(model_id, registered=True)
    if not path.exists():
        return {"valid": False, "error": "model_not_registered"}
    card = json.loads(path.read_text())
    if card.get("status") != "validated" or card.get("task") != task:
        return {"valid": False, "error": "incompatible_or_unvalidated_model"}
    predictions = []
    for row in rows(input_csv):
        value = float(np.array([1, float(row["x1"]), float(row["x2"])]) @ np.array(card["coefficients"]))
        predictions.append({"id": row["id"], "prediction": value})
    return {"valid": True, "predictions": predictions}


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
