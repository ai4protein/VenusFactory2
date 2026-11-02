"""Bounded MCP surface: create candidates or reuse read-only validated capabilities.

This entry point is bundled with protein_model.py; it has no controller import,
publication tool, hidden labels, external provider helper or shell execution.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastmcp import FastMCP
from protein_model import digest, fit_candidate, predict, validate_rows

mcp = FastMCP("VenusFactoryProteinModels")
ROOT = Path(os.environ.get("VENUS_RSI_WORKSPACE", "/workspace")).resolve()
CAPABILITIES = Path(__file__).resolve().parent


def workspace_file(name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        path = path.relative_to(ROOT)
    if not path.parts or ".." in path.parts:
        raise ValueError("invalid workspace path")
    current = ROOT
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("symlink outside artifact contract")
    current.resolve().relative_to(ROOT)
    return current


def task() -> dict:
    # Trusted task metadata lives in the read-only capability bundle.
    return json.loads((CAPABILITIES / "task.json").read_text())


def frozen_rows(name: str) -> list[dict]:
    rows = json.loads(workspace_file(name).read_text())
    if digest(rows) != task()["inputs"][name]:
        raise ValueError("task input changed")
    return rows


def write_output(name: str, value):
    dest = workspace_file(name)
    if dest.exists():
        raise ValueError("artifact already exists")
    dest.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")


@mcp.tool()
def inspect_protein_task() -> dict:
    """Inspect assay scope, aligned mutation sites and training-only summary. No hidden labels."""
    info = task()
    result = {
        "phase": info["phase"],
        "scope": info["scope"],
        "inputs": list(info["inputs"]),
        "development_type": "configured_task_specific_model",
        "supported_encodings": ["onehot", "onehot_pairwise"],
        "maximum_candidates": 1,
    }
    if info["phase"] == "develop":
        rows = frozen_rows("training.json")
        validate_rows(rows, info["scope"], labels=True)
        y = [r["target"] for r in rows]
        result["training"] = {
            "rows": len(rows),
            "target_min": min(y),
            "target_max": max(y),
            "target_mean": sum(y) / len(y),
        }
    return result


@mcp.tool()
def develop_protein_model(encoding: str, alphas: list[float]) -> dict:
    """Train one candidate with chosen encoding and ridge alpha grid, using 5-fold training-only CV. Writes candidate.json; does NOT validate or register it."""
    info = task()
    if info["phase"] != "develop" or workspace_file("candidate.json").exists():
        raise ValueError("candidate creation is unavailable or budget exhausted")
    model = fit_candidate(frozen_rows("training.json"), info["scope"], {"encoding": encoding, "alphas": alphas})
    write_output("candidate.json", model)
    return {
        "artifact": "candidate.json",
        "status": "candidate",
        "registered": False,
        "selected_alpha": model["selected_alpha"],
        "cv_rmse": model["cv_rmse"],
        "training_rows": model["training_rows"],
        "model_sha256": digest(model),
    }


@mcp.tool()
def discover_protein_models() -> dict:
    """Find read-only validated capabilities compatible with this task's protein, assay, units and aligned sites."""
    info = task()
    packet_file = CAPABILITIES / "registered/capability.json"
    if info["phase"] != "reuse" or not packet_file.exists():
        return {"models": []}
    packet = json.loads(packet_file.read_text())
    if (
        digest(packet) != info["capability_sha256"]
        or digest(packet["model"]) != packet["model_sha256"]
        or packet["model"]["scope"] != info["scope"]
    ):
        raise ValueError("capability integrity or applicability mismatch")
    return {
        "models": [
            {
                "model_id": packet["model_id"],
                "scope": packet["model"]["scope"],
                "plugin_id": packet["model"]["plugin_id"],
                "status": "registered",
                "validation": packet["assessment"]["metrics"],
            }
        ]
    }


@mcp.tool()
def predict_with_protein_model(model_id: str) -> dict:
    """Reuse a discovered validated model on the new task inputs, writing predictions.json with sample IDs."""
    catalog = discover_protein_models()
    if not any(x["model_id"] == model_id for x in catalog["models"]):
        raise ValueError("no compatible validated model")
    packet = json.loads((CAPABILITIES / "registered/capability.json").read_text())
    rows = frozen_rows("inputs.json")
    predictions = predict(packet["model"], rows, task()["scope"])
    write_output("predictions.json", predictions)
    return {
        "artifact": "predictions.json",
        "model_id": model_id,
        "prediction_rows": len(predictions),
        "prediction_sha256": digest(predictions),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
