"""Promote source-audited POISK cohorts to native Scientist model tasks."""

from __future__ import annotations

import json
from pathlib import Path

from agent.rsi import generated_antibody as api
from agent.rsi import model_tasks
from agent.rsi import poisk_pair_encoder as encoder
from .poisk_source import RECORD, require
from agent.rsi.engine.storage import atomic_json, sha256_file


def binding(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}


def prepare(source, feasibility, output):
    source, feasibility, output = (Path(p).absolute() for p in (source, feasibility, output))
    require(
        not output.exists() and not any(p.is_symlink() for p in [output, *output.parents]),
        "New ordinary task directory required",
    )
    source_manifest = json.loads((source / "manifest.json").read_text())
    require(sha256_file(source / "observations.json") == source_manifest["observations_sha256"], "Source changed")
    old = json.loads((feasibility / "manifest.json").read_text())
    require(old["source_manifest_sha256"] == sha256_file(source / "manifest.json"), "Feasibility source differs")
    source_rows = json.loads((source / "observations.json").read_text())
    lookup = {r["id"]: r for r in source_rows}
    require(len(lookup) == len(source_rows), "Repeated source observation")
    groups = json.loads((feasibility / "group_evidence.json").read_text())["groups"]
    output.mkdir(parents=True)
    encoder_path = output / "poisk_pair_encoder.py"
    encoder_path.write_bytes(Path(encoder.__file__).read_bytes())
    tasks, excluded = [], []
    for original in old["tasks"]:
        task_id = original["id"]
        old_path = feasibility / task_id / "task.json"
        require(json.loads(old_path.read_text()) == original, "Feasibility task declaration changed")
        scope = {
            "protein": original["scope"]["target_accession"],
            "assay": "POISK source-reported " + original["scope"]["method"] + " antibody measurement",
            "target": api.ENDPOINTS[original["scope"]["endpoint"]],
            "unit": "negative_log10_molar",
            "context": {
                "dataset": "POISK v1.1",
                "organism": "Homo sapiens",
                "antibody_format": original["scope"]["antibody_type"],
                "measurement_method": original["scope"]["method"],
                "reported_endpoint": original["scope"]["endpoint"],
                "condition": original["scope"]["condition"],
            },
            "covariates": encoder.covariates(),
            "representation": {
                "name": encoder.NAME,
                "dimensions": len(encoder.covariates()),
                "encoder_sha256": sha256_file(encoder_path),
            },
            "interface_version": api.PLUGIN_ID,
            "grouping": {"level": "antibody_patent_component", "evidence_sha256": "0" * 64},
        }
        parts = {}
        for phase in model_tasks.PHASES:
            old_rows = json.loads(model_tasks.bound_file(original["data"][phase]).read_text())
            rows = []
            for old_row in old_rows:
                row = lookup[old_row["id"]]
                require(
                    old_row["target"] == row["target"] and old_row["group"] == groups[row["id"]],
                    "Original observations or grouping changed",
                )
                native = {
                    k: row[k]
                    for k in ("id", "antibody", "heavy", "light", "target_sequence_sha256", "patents", "target")
                }
                native.update(protein=row["target_accession"], group=old_row["group"])
                native["values"] = encoder.encode(native)
                require(native["values"] == old_row["values"], "Descriptor map differs from feasibility controls")
                rows.append(native)
            parts[phase] = rows
            api.validate_rows(rows, scope, labels=True)
        api.require_disjoint_parts(parts, scope)
        try:
            api.development_split(parts["training"])
        except ValueError as exc:
            excluded.append({"id": task_id, "reason": str(exc)})
            continue
        folder = output / task_id
        folder.mkdir()
        refs = [
            binding(source / "observations.json"),
            binding(source / "manifest.json"),
            binding(feasibility / "manifest.json"),
            binding(feasibility / "group_evidence.json"),
            binding(old_path),
        ]
        evidence = {
            "schema": "antibody_groups.v1",
            "level": scope["grouping"]["level"],
            "study_doi": RECORD,
            "sources": refs,
            "source_observations": refs[0],
            "observations": {
                r["id"]: {"group": r["group"], "phase": phase, "input_sha256": api.input_digest(r)}
                for phase, rows in parts.items()
                for r in rows
            },
        }
        atomic_json(folder / "groups.json", evidence)
        scope["grouping"]["evidence_sha256"] = sha256_file(folder / "groups.json")
        data = {}
        for phase, rows in parts.items():
            atomic_json(folder / f"{phase}.json", rows)
            data[phase] = binding(folder / f"{phase}.json")
        task = {
            "schema": 1,
            "id": task_id,
            "stage": "binding",
            "plugin_id": api.PLUGIN_ID,
            "scope": scope,
            "paper": {"title": "POISK v1.1 antibody patent dataset", "doi": RECORD},
            "references": [*refs, binding(encoder_path), binding(folder / "groups.json")],
            "data": data,
            "minimum_skill": 0.0,
        }
        model_tasks.load_task(task)
        atomic_json(folder / "task.json", task)
        tasks.append(task)
    atomic_json(output / "tasks.json", tasks)
    report = {
        "schema": "poisk_native_tasks.v1",
        "status": "prepared_not_started",
        "plugin_id": api.PLUGIN_ID,
        "tasks": tasks,
        "excluded": excluded,
        "task_count": len(tasks),
        "source_manifest_sha256": sha256_file(source / "manifest.json"),
        "original_feasibility_manifest_sha256": sha256_file(feasibility / "manifest.json"),
        "split_policy": "Existing source-audited partitions retained; no label-driven repartitioning or task selection",
        "shared_execution": "Requires shared_model_bundle.admit to verify cross-task partitions; separate tasks are not automatically safe to co-prompt",
        "external_model_calls": 0,
        "new_autonomous_models": 0,
        "final_labels_scored": False,
    }
    atomic_json(output / "manifest.json", report)
    return report
