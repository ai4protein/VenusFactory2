"""Host-admitted protein modelling tasks for the shared Scientist lifecycle."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import sha256_file

PLUGINS = {
    "generated_python.v1": (10, 600),
    "generated_sequence.v1": (10, 600),
    "generated_features.v1": (10, 600),
    "generated_context.v1": (10, 600),
    "generated_context.v2": (10, 600),
    "generated_context.v3": (10, 600),
    "generated_assay.v1": (10, 600),
    "generated_antibody.v1": (10, 600),
    "generated_classifier.v1": (20, 50000),
    "generated_classifier.large.v1": (20, 150000),
}
PHASES = ("training", "validation", "future")
STAGES = {
    "function_annotation",
    "activity",
    "stability",
    "solubility",
    "expression",
    "substrate_specificity",
    "kinetics",
    "developability",
    "binding",
    "fitness",
    "other",
}


def ordinary(path):
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError("Task assets require absolute, non-symlinked paths")
    if not path.is_file():
        raise ValueError("Missing task asset")
    return path


def bound_file(row):
    if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
        raise ValueError("Task files require a path and SHA256")
    if not isinstance(row["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
        raise ValueError("Invalid task asset hash")
    path = ordinary(row["path"])
    if sha256_file(path) != row["sha256"]:
        raise ValueError("Task asset changed")
    return path


def load_task(definition):
    """Validate all complete partitions before allocating a goal or model call."""
    if not isinstance(definition, dict) or set(definition) != {
        "schema",
        "id",
        "stage",
        "plugin_id",
        "scope",
        "paper",
        "references",
        "data",
        "minimum_skill",
    }:
        raise ValueError("Invalid model task definition fields")
    if type(definition["schema"]) is not int or definition["schema"] != 1:
        raise ValueError("Unsupported model task schema")
    if not isinstance(definition["id"], str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,119}", definition["id"]):
        raise ValueError("Invalid model task identity")
    if definition["stage"] not in STAGES or definition["plugin_id"] not in PLUGINS:
        raise ValueError("Unknown modelling stage or generated-model interface")
    paper = definition["paper"]
    if (
        not isinstance(paper, dict)
        or set(paper) != {"title", "doi"}
        or any(not isinstance(v, str) or not v.strip() for v in paper.values())
    ):
        raise ValueError("A paper title and persistent reference are required")
    minimum = definition["minimum_skill"]
    if type(minimum) not in (int, float) or not math.isfinite(minimum) or not 0 <= minimum < 1:
        raise ValueError("Invalid predeclared acceptance threshold")
    refs = definition["references"]
    if not isinstance(refs, list) or not 1 <= len(refs) <= 100:
        raise ValueError("Source references must be explicitly bound")
    referenced = [str(bound_file(row)) for row in refs]
    if len(set(referenced)) != len(referenced):
        raise ValueError("Repeated task reference")
    if not isinstance(definition["data"], dict) or set(definition["data"]) != set(PHASES):
        raise ValueError("Training, acceptance and future partitions are required")
    plugin = get_model_plugin(definition["plugin_id"])
    contextual = plugin.id in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }
    data, ids, sequences = {}, set(), set()
    for phase in PHASES:
        path = bound_file(definition["data"][phase])
        if path.stat().st_size > 128 * 1024**2:
            raise ValueError("Model task partition exceeds the admitted file size")
        rows = json.loads(path.read_text())
        plugin.validate_rows(rows, definition["scope"], labels=True)
        if len(rows) < 2:
            raise ValueError("Every partition needs at least two observations")
        new_ids = {r["id"] for r in rows}
        new_sequences = {r["group"] for r in rows} if contextual else {r["sequence"] for r in rows}
        if ids & new_ids or sequences & new_sequences:
            raise ValueError(
                "Model task partitions overlap by identity or "
                + ("independent group" if contextual else "exact sequence")
            )
        ids.update(new_ids)
        sequences.update(new_sequences)
        data[phase] = rows
    if contextual:
        context_group_evidence(definition, data)
    lo, hi = PLUGINS[plugin.id]
    if not lo <= len(data["training"]) <= hi:
        raise ValueError("Training allocation exceeds the selected model interface")
    if plugin.id.endswith("large.v1") and any(len(data[p]) > 60000 for p in ("validation", "future")):
        raise ValueError("Evaluation allocation exceeds the selected prediction interface")
    if "class_labels" in definition["scope"]:
        if {r["target"] for r in data["training"]} != set(range(len(definition["scope"]["class_labels"]))):
            raise ValueError("Training must contain every declared class")
    for key in ("protein", "assay", "target", "unit"):
        if not isinstance(definition["scope"].get(key), str) or not definition["scope"][key].strip():
            raise ValueError("Protein, assay, endpoint and unit must be specified")
    return data


def context_group_evidence(definition, data):
    """Bind host-reviewed sample grouping to source files, not model assertions."""
    if definition["plugin_id"] in {"generated_assay.v1", "generated_antibody.v1"}:
        from agent.rsi.engine.context_interfaces import module_for

        return module_for(definition["plugin_id"]).validate_task_evidence(definition, data, bound_file)
    from agent.rsi.engine.generated_context import development_split

    scope = definition["scope"]
    matches = [r for r in definition["references"] if r["sha256"] == scope["grouping"]["evidence_sha256"]]
    if len(matches) != 1:
        raise ValueError("Context grouping evidence must be uniquely bound in task references")
    evidence = json.loads(bound_file(matches[0]).read_text())
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"schema", "level", "study_doi", "sources", "observations"}
        or evidence["schema"] != "context_groups.v1"
        or evidence["level"] != scope["grouping"]["level"]
        or evidence["study_doi"] != definition["paper"]["doi"]
        or not isinstance(evidence["sources"], list)
        or not evidence["sources"]
        or not isinstance(evidence["observations"], dict)
    ):
        raise ValueError("Invalid source-bound context grouping evidence")
    for source in evidence["sources"]:
        if source == matches[0] or source not in definition["references"]:
            raise ValueError("Grouping source must be a separately bound task reference")
        bound_file(source)
    expected = {row["id"]: {"group": row["group"], "phase": phase} for phase, rows in data.items() for row in rows}
    if evidence["observations"] != expected:
        raise ValueError("Context group assignments differ from source-bound evidence")
    development_split(data["training"])
    return evidence


def read_task(path):
    path = ordinary(path)
    if path.stat().st_size > 131072:
        raise ValueError("Model task declaration is too large")
    definition = json.loads(path.read_text())
    return definition, load_task(definition)


def admission(definition):
    return {"kind": "scientist_model_task.v1", "definition_sha256": digest(definition)}


def policy_for_task(definition, data):
    plugin = get_model_plugin(definition["plugin_id"])
    return {
        "schema": 1,
        "scope": definition["scope"],
        "plugin_id": plugin.id,
        "plugin_sha256": plugin.source_sha256,
        "training_sha256": digest(data["training"]),
        "validation_sha256": digest(data["validation"]),
        "minimum_skill": definition["minimum_skill"],
        "max_candidates": 1,
    }


def source_and_split(definition):
    if definition["plugin_id"] in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        return (
            {"kind": "host_admitted_model_task", "paper": definition["paper"], "references": definition["references"]},
            {
                "kind": "predeclared_independent_groups",
                "data": definition["data"],
                "grouping": definition["scope"]["grouping"],
                "observation_and_group_disjoint": True,
            },
        )
    return (
        {"kind": "host_admitted_model_task", "paper": definition["paper"], "references": definition["references"]},
        {
            "kind": "predeclared_partitions",
            "data": definition["data"],
            "identity_and_exact_sequence_disjoint": True,
            "homology_independence_established": False,
        },
    )


def inference_inputs(rows, plugin_id):
    if plugin_id in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        from agent.rsi.engine.context_interfaces import module_for

        return module_for(plugin_id).inference_inputs(rows)
    fields = ("id", "sequence", "features") if plugin_id == "generated_features.v1" else ("id", "sequence")
    return [{name: row[name] for name in fields} for row in rows]
