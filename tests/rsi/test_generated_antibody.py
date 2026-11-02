"""Synthetic paired-chain numerical tests, not autonomous research outcomes."""

import hashlib
import random
from copy import deepcopy
from pathlib import Path

import pytest

from agent.rsi import generated_antibody as api
from agent.rsi import model_tasks
from agent.rsi import poisk_pair_encoder as encoder
from agent.rsi.poisk_tasks import binding, prepare
from agent.rsi.protein_model import digest
from agent.rsi.storage import atomic_json, sha256_file

CODE = """def fit(features, targets, seed):
    x = [row[0] for row in features]
    xm = sum(x)/len(x)
    ym = sum(targets)/len(targets)
    slope = sum((a-xm)*(b-ym) for a,b in zip(x,targets))/sum((a-xm)**2 for a in x)
    return {"slope": slope, "intercept": ym-slope*xm}
def predict(model, features):
    return [model["intercept"]+model["slope"]*row[0] for row in features]
"""


def make_source(root, accession="P00533", kind="Conventional"):
    root = Path(root)
    source, feasibility = root / "source", root / "feasibility"
    source.mkdir(parents=True)
    feasibility.mkdir()
    rng = random.Random(881)
    observations, parts, groups, assignment = [], {}, {}, {}
    offset = 0
    for phase, number in [("training", 8), ("validation", 3), ("future", 3)]:
        rows = []
        for g in range(offset, offset + number):
            group = f"group_{g}"
            assignment[group] = phase
            for sample in range(8):
                heavy = "".join(rng.choices(encoder.AA, k=100))
                light = "".join(rng.choices(encoder.AA, k=90)) if kind == "Conventional" else ""
                row = {
                    "id": f"{accession}_{g}_{sample}",
                    "heavy": heavy,
                    "light": light,
                    "antibody": digest([heavy, light]),
                    "target_accession": accession,
                    "target_sequence_sha256": hashlib.sha256(accession.encode()).hexdigest(),
                    "patents": [str(100000 + g)],
                    "source_records": [f"record_{accession}_{g}_{sample}"],
                    "target": 2 + 5 * heavy.count("A") / len(heavy),
                    "antibody_type": kind,
                    "method": "SPR",
                    "endpoint": "reported_molar_concentration",
                    "condition": "",
                }
                observations.append(row)
                groups[row["id"]] = group
                rows.append(
                    {
                        "id": row["id"],
                        "group": group,
                        "target": row["target"],
                        "values": encoder.encode({**row, "protein": accession}),
                        "source_records": row["source_records"],
                    }
                )
        parts[phase] = rows
        offset += number
    atomic_json(source / "observations.json", observations)
    atomic_json(source / "manifest.json", {"observations_sha256": sha256_file(source / "observations.json")})
    task_id = "poisk_fixture_" + accession.lower()
    folder = feasibility / task_id
    folder.mkdir()
    data = {}
    for phase, rows in parts.items():
        atomic_json(folder / f"{phase}.json", rows)
        data[phase] = binding(folder / f"{phase}.json")
    original = {
        "id": task_id,
        "scope": {
            "target_accession": accession,
            "method": "SPR",
            "antibody_type": kind,
            "endpoint": "reported_molar_concentration",
            "condition": "unreported_in_source",
        },
        "data": data,
        "split_assignment": assignment,
    }
    atomic_json(folder / "task.json", original)
    atomic_json(
        feasibility / "manifest.json",
        {"source_manifest_sha256": sha256_file(source / "manifest.json"), "tasks": [original]},
    )
    atomic_json(feasibility / "group_evidence.json", {"groups": groups})
    return source, feasibility


def make_task(root, accession="P00533", kind="Conventional"):
    source, feasibility = make_source(root, accession, kind)
    report = prepare(source, feasibility, Path(root) / "native")
    assert report["task_count"] == 1 and report["new_autonomous_models"] == 0
    task = report["tasks"][0]
    return task, model_tasks.load_task(task)


@pytest.mark.parametrize("kind", ["Conventional", "VHH"])
def test_paired_identity_fitting_and_fresh_prediction(tmp_path, kind):
    task, parts = make_task(tmp_path, kind=kind)
    model = api.fit_candidate(parts["training"], task["scope"], {"source_code": CODE, "seed": 19})
    metrics, predictions = api.evaluate_with_predictions(model, parts["validation"], task["scope"])
    assert metrics[api.ACCEPTANCE_METRIC] > 0.999999
    assert api.prediction_rows_match(predictions, parts["validation"])
    assert all("target" not in r for r in api.inference_inputs(parts["future"]))
    with pytest.raises(ValueError, match="overlap"):
        api.evaluate(model, parts["training"], task["scope"])


@pytest.mark.parametrize("attack", ["descriptors", "pair", "target", "format", "patent", "unit", "endpoint"])
def test_malformed_or_mislabeled_antibody_rows_rejected(tmp_path, attack):
    task, parts = make_task(tmp_path)
    scope, rows = deepcopy(task["scope"]), deepcopy(parts["training"])
    if attack == "descriptors":
        rows[0]["values"][0] += 0.01
    elif attack == "pair":
        rows[0]["heavy"] = rows[0]["heavy"][::-1]
    elif attack == "target":
        rows[0]["protein"] = "P04626"
    elif attack == "format":
        rows[0]["light"] = ""
    elif attack == "patent":
        rows[0]["patents"] = ["unknown"]
    elif attack == "unit":
        scope["unit"] = "nM"
    else:
        scope["target"] = "negative_log10_KD_molar"
    with pytest.raises(ValueError):
        api.validate_rows(rows, scope, labels=True)


@pytest.mark.parametrize("attack", ["patent", "chain", "near_clone"])
def test_changed_group_names_do_not_hide_dependence(tmp_path, attack):
    task, parts = make_task(tmp_path)
    row = deepcopy(parts["validation"][0])
    train = parts["training"][0]
    row["group"] = "renamed_group"
    if attack == "patent":
        row["patents"] = train["patents"]
    elif attack == "chain":
        row["heavy"] = train["heavy"]
    else:
        for chain in ("heavy", "light"):
            sequence = train[chain]
            row[chain] = ("A" if sequence[0] != "A" else "C") + sequence[1:]
    row["antibody"] = digest([row["heavy"], row["light"]])
    row["values"] = encoder.encode(row)
    api.validate_rows([row], task["scope"], labels=True)
    with pytest.raises(ValueError, match="crosses partitions"):
        api.require_disjoint_parts({"training": parts["training"], "other": [row]}, task["scope"])


def test_source_target_cannot_be_replaced_with_a_new_bound_label(tmp_path):
    task, parts = make_task(tmp_path)
    path = Path(task["data"]["validation"]["path"])
    parts["validation"][0]["target"] += 1
    atomic_json(path, parts["validation"])
    task["data"]["validation"] = binding(path)
    with pytest.raises(ValueError, match="differs from source"):
        model_tasks.load_task(task)


def test_encoder_matches_original_feasibility_representation(tmp_path):
    from agent.rsi.poisk_training import encode

    task, parts = make_task(tmp_path)
    for row in parts["training"]:
        assert encoder.encode(row) == encode({**row, "target_accession": row["protein"]})
    assert len(encoder.covariates()) == 118
    train, feedback = api.development_split(parts["training"])
    assert not {r["group"] for r in train} & {r["group"] for r in feedback}
    assert len(train) >= 8 and len(feedback) >= 2


def test_five_fixed_controls_keep_native_input_identity(tmp_path):
    from agent.rsi.antibody_baselines import run_controls

    task, parts = make_task(tmp_path)
    report = run_controls(task)
    assert len(report["outcomes"]) == 5 and report["new_autonomous_models"] == 0
    assert report["future_scored"] is False
    for result in report["outcomes"]:
        assert result["status"] == "completed"
        assert [p["inputs_sha256"] for p in result["predictions"]] == [api.input_digest(r) for r in parts["validation"]]


def test_development_partition_does_not_use_label_magnitudes(tmp_path):
    task, parts = make_task(tmp_path)
    rows = deepcopy(parts["training"])
    before = api.development_split(rows)
    for row in rows:
        row["target"] = 18 - row["target"]
    after = api.development_split(rows)
    assert [[r["id"] for r in part] for part in before] == [[r["id"] for r in part] for part in after]


def test_task_promotion_keeps_original_partitions_and_rejects_tampered_source(tmp_path):
    import json

    source, feasibility = make_source(tmp_path)
    result = prepare(source, feasibility, tmp_path / "native")
    task = result["tasks"][0]
    original = json.loads((feasibility / "manifest.json").read_text())["tasks"][0]
    for phase, rows in model_tasks.load_task(task).items():
        old = json.loads(Path(original["data"][phase]["path"]).read_text())
        assert [(r["id"], r["group"], r["target"], r["values"]) for r in rows] == [
            (r["id"], r["group"], r["target"], r["values"]) for r in old
        ]
    path = source / "observations.json"
    observations = json.loads(path.read_text())
    observations[0]["target"] += 1
    atomic_json(path, observations)
    with pytest.raises(ValueError, match="Source changed"):
        prepare(source, feasibility, tmp_path / "native_rejected")
    assert not (tmp_path / "native_rejected").exists()
