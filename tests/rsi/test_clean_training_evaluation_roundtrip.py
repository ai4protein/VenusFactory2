"""Reduced synthetic fixtures exercise artifact replay, not biological execution."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from agent.rsi import clean_annotation as annotation
from agent.rsi import clean_training_evaluation as e
from agent.rsi import clean_training_inputs as inputs


def native_projection(state, means):
    h = torch.from_numpy(means)
    for i in (1, 2, 3):
        h = torch.nn.functional.linear(h, state[f"fc{i}.weight"], state[f"fc{i}.bias"])
        if i < 3:
            h = torch.relu(
                torch.nn.functional.layer_norm(h, (512,), state[f"ln{i}.weight"], state[f"ln{i}.bias"], 1e-5)
            )
    return h


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    torch.manual_seed(42)
    generator = np.random.default_rng(42)
    output, source, training, assets = [tmp_path / n for n in ("output", "input", "training", "assets")]
    for path in (output, source, training, assets / "source/app/data"):
        path.mkdir(parents=True)
    labels = ["1.1.1." + str(i) for i in range(12)]
    records = {
        "train": [
            {"id": "T" + str(i), "key": "T" + str(i), "sequence": "A", "tokens": 1, "labels": [label]}
            for i, label in enumerate(labels)
        ]
    }
    records.update(
        {
            name: [
                {"id": name + str(i), "key": name + str(i), "sequence": "D", "tokens": 1, "labels": [labels[i]]}
                for i in range(2)
            ]
            for name in e.COHORTS
        }
    )
    means = {name: generator.standard_normal((len(rows), 1280), dtype=np.float32) for name, rows in records.items()}
    with (assets / "source/app/data/split10.csv").open("w") as stream:
        stream.write("Entry\tEC number\tSequence\n")
        for row in records["train"]:
            stream.write(row["id"] + "\t" + row["labels"][0] + "\tA\n")
    inputs.save(source / "plan.json", {})
    monkeypatch.setattr(inputs, "verify_assets", lambda path: {})
    monkeypatch.setattr(e, "admit_inputs", lambda path: {"synthetic_fixture": True})
    monkeypatch.setattr(
        e, "admit_training", lambda root, assets, algorithm: {"synthetic_fixture": True, "algorithm": algorithm}
    )
    monkeypatch.setattr(e, "load_means", lambda root, plan, cohort: (records[cohort], means[cohort]))
    monkeypatch.setattr(e, "COHORTS", dict.fromkeys(e.COHORTS, 2))
    monkeypatch.setattr(
        e,
        "ALLOCATION",
        {"models": list(e.SCHEDULES), "cohorts": e.COHORTS, "sequence_records": 6, "prediction_records": 12},
    )
    implementation = {}
    runner = Path(e.__file__).resolve().parents[4] / "scripts/rsi/cases/run_clean_training_evaluation.py"
    for path in (Path(e.__file__), Path(inputs.__file__), Path(annotation.__file__), runner):
        shutil.copy2(path, output / path.name)
        implementation[path.name] = inputs.sha(path)
    inputs.save(output / "protocol.json", e.evaluation_protocol(implementation))
    inputs.save(output / "input_admission.json", {"synthetic_fixture": True})
    inputs.save(
        output / "training_admission.json", {a: {"synthetic_fixture": True, "algorithm": a} for a in e.SCHEDULES}
    )
    report = {
        "status": "computed_pending_audit",
        "stage": "independent_audit",
        "allocation": e.ALLOCATION,
        "external_model_calls": 0,
        "training_completed": True,
        "evaluation_completed": False,
        "execution_origin": "controller_executed_author_training_and_numerically_audited_evaluation",
        "results": {},
        "failures": {},
        **e.LIMITS,
    }
    for algorithm, schedule in e.SCHEDULES.items():
        state = {}
        for i, (n_in, n_out) in enumerate(((1280, 512), (512, 512), (512, schedule["out_dim"])), 1):
            layer = torch.nn.Linear(n_in, n_out)
            state.update({f"fc{i}.weight": layer.weight.detach(), f"fc{i}.bias": layer.bias.detach()})
            if i < 3:
                state.update({f"ln{i}.weight": torch.ones(512), f"ln{i}.bias": torch.zeros(512)})
        (training / algorithm).mkdir()
        torch.save(state, training / algorithm / "final.pth")
        center = native_projection(state, means["train"])
        np.save(output / (algorithm + ".train_projection.npy"), center.numpy(), allow_pickle=False)
        np.save(output / (algorithm + ".centers.npy"), center.numpy(), allow_pickle=False)
        inputs.save(output / (algorithm + ".classes.json"), labels)
        for cohort in e.COHORTS:
            key = algorithm + "." + cohort
            projected = native_projection(state, means[cohort])
            distances = torch.stack([(center - row).norm(dim=1) for row in projected]).numpy()
            np.save(output / (key + ".projection.npy"), projected.numpy(), allow_pickle=False)
            np.save(output / (key + ".distances.npy"), distances, allow_pickle=False)
            inputs.save(output / (key + ".ids.json"), e.identity_records(records[cohort]))
            predictions = annotation.select_labels(distances, labels)
            with (output / (key + "_maxsep.csv")).open("w", newline="") as stream:
                writer = csv.writer(stream)
                for i, row in enumerate(predictions):
                    writer.writerow(
                        [
                            records[cohort][i]["id"],
                            *[f"EC:{label}/{distances[i, labels.index(label)]:.4f}" for label in row],
                        ]
                    )
            truth = [r["labels"] for r in records[cohort]]
            report["results"][key] = {
                "rows": 2,
                "train_id_overlap": 0,
                "train_exact_sequence_overlap": 0,
                "full_union_metrics": e.sklearn_metrics(truth, predictions),
                "truth_only_label_universe_metrics": e.sklearn_metrics(truth, predictions, truth_only=True),
            }
    report["artifact_sha256"] = {name: inputs.sha(output / name) for name in e.artifact_names()}
    inputs.save(output / "report.json", report)
    return output, source, training, assets, report


def test_artifact_roundtrip_and_completed_audit_binding(artifacts):
    output, source, training, assets, report = artifacts
    audit = e.verify_output(output, source, training, assets)
    assert audit["status"] == "passed" and len(audit["allocations"]) == 6
    inputs.save(output / "independent_audit.json", audit)
    report.update(
        status="completed",
        stage="audited_evaluation",
        evaluation_completed=True,
        independent_audit_sha256=inputs.sha(output / "independent_audit.json"),
    )
    inputs.save(output / "report.json", report)
    assert e.verify_output(output, source, training, assets) == audit
    altered = json.loads((output / "independent_audit.json").read_text())
    altered["status"] = "failed"
    inputs.save(output / "independent_audit.json", altered)
    report["independent_audit_sha256"] = inputs.sha(output / "independent_audit.json")
    inputs.save(output / "report.json", report)
    with pytest.raises(ValueError, match="audit binding"):
        e.verify_output(output, source, training, assets)


@pytest.mark.parametrize("mutation", ["distance", "metrics", "metadata"])
def test_rehashed_numerical_or_result_tampering_retains_failed_allocations(artifacts, mutation):
    output, source, training, assets, report = artifacts
    key = "triplet.new"
    if mutation == "distance":
        name = key + ".distances.npy"
        changed = np.load(output / name, allow_pickle=False)
        changed += 1.0
        np.save(output / name, changed, allow_pickle=False)
        report["artifact_sha256"][name] = inputs.sha(output / name)
    elif mutation == "metrics":
        report["results"][key]["full_union_metrics"]["micro_f1"] = 9.0
    else:
        report["results"][key]["train_id_overlap"] = 1
    inputs.save(output / "report.json", report)
    audit = e.verify_output(output, source, training, assets)
    assert audit["status"] == "failed" and len(audit["allocations"]) == 6
    assert audit["allocations"][key]["status"] == "failed"
    assert audit["allocations"]["supconH.new"]["status"] == "passed"


@pytest.mark.parametrize(
    "mutation", ["source", "identity", "coverage", "autonomous", "evaluation_flag", "training_flag", "origin", "stage"]
)
def test_evaluation_provenance_tampering_fails(artifacts, mutation):
    output, source, training, assets, report = artifacts
    if mutation == "source":
        name = "clean_training_evaluation.py"
        (output / name).write_text("different evaluator")
        report["artifact_sha256"][name] = inputs.sha(output / name)
        protocol = json.loads((output / "protocol.json").read_text())
        protocol["implementation"][name] = report["artifact_sha256"][name]
        inputs.save(output / "protocol.json", protocol)
        report["artifact_sha256"]["protocol.json"] = inputs.sha(output / "protocol.json")
    elif mutation == "identity":
        name = "triplet.new.ids.json"
        ids = json.loads((output / name).read_text())
        ids.reverse()
        inputs.save(output / name, ids)
        report["artifact_sha256"][name] = inputs.sha(output / name)
    elif mutation == "coverage":
        report["artifact_sha256"].pop("supconH.new_maxsep.csv")
    elif mutation == "autonomous":
        report["autonomous_model_developed"] = True
    elif mutation == "evaluation_flag":
        report["evaluation_completed"] = True
    elif mutation == "training_flag":
        report["training_completed"] = False
    elif mutation == "origin":
        report["execution_origin"] = "untrusted"
    else:
        report["stage"] = "done"
    inputs.save(output / "report.json", report)
    with pytest.raises(ValueError):
        e.verify_output(output, source, training, assets)
