"""Read-only evidence checks for terminal matched context-history studies."""

import hashlib
import json
import math
from pathlib import Path

from .context_baseline_refit import verify_controls
from .matched_history_analysis import METRICS, require, summarize, validate_design
from agent.rsi.campaigns.matched_history_operations import collect_unit, summarize_operations
from .protein_model import digest
from agent.rsi.campaigns.scientist_model_campaign import process_live
from .shared_sequence_evaluation import require_terminal


class Evidence:
    """Pin ordinary files under one research repository and detect later changes."""

    def __init__(self, repository):
        self.repository = Path(repository).absolute()
        self.bindings = {}

    def path(self, path, *, inside=None):
        path = Path(path).absolute()
        require(".." not in path.parts, "Parent traversal is unsupported")
        require(
            path.is_relative_to(Path(inside or self.repository).absolute()), "Evidence path leaves allowed directory"
        )
        require(not any(p.is_symlink() for p in [path, *path.parents]), "Symlinked evidence is unsupported")
        require(
            not any(
                p == "private_home" or p.startswith(".env") or p == "20261014_improver_revision_glm52_v1"
                for p in path.parts
            ),
            "Excluded evidence path",
        )
        return path

    def read_bytes(self, path, expected=None, *, inside=None):
        path = self.path(path, inside=inside)
        require(path.is_file(), "Evidence file missing: " + str(path))
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        require(expected is None or sha == expected, "Frozen evidence changed: " + str(path))
        require(str(path) not in self.bindings or self.bindings[str(path)] == sha, "Evidence changed during analysis")
        self.bindings[str(path)] = sha
        return data

    def read(self, path, expected=None, *, inside=None):
        return json.loads(self.read_bytes(path, expected, inside=inside))

    def check_bindings(self, bindings):
        for path, sha in bindings.items():
            self.read_bytes(path, sha)

    def unchanged(self):
        self.check_bindings(dict(self.bindings))


def terminal_study(evidence, study, launch_sha256):
    """Global barrier runs before opening reuse, evaluation or label files."""
    study = evidence.path(study)
    launch = evidence.read(study / "launch_basis.json", launch_sha256)
    design = evidence.read(study / "study_design.json", launch["study_design_sha256"])
    require(launch["final_protocol_sha256"] == design["final_protocol_sha256"], "Final protocol pin differs")
    identity = evidence.read(study / "runner_identity.json")
    state = evidence.read(study / "runner_state.json")
    require(identity["study"] == str(study), "Study runner identity differs")
    require(not process_live(identity), "Global terminal barrier: study runner is live")
    require(state["status"] == "terminal_pending_independent_reuse", "Global terminal barrier: study not terminal")
    completed = state["completed_units"]
    unit_ids = {u["id"] for u in design["units"]}
    require(
        len(completed) == len(unit_ids) and {u["unit"] for u in completed} == unit_ids,
        "Global terminal barrier: incomplete unit coverage",
    )
    require(
        all(u["status"] in {"completed", "needs_tasks", "budget_exhausted"} for u in completed),
        "Failed unit cannot be silently omitted",
    )
    plans, contexts = {}, {}
    for unit in design["units"]:
        folder = evidence.path(study / unit["id"], inside=study)
        require(folder.parent == study, "Unit must be a direct study child")
        require(
            unit["campaign"] == str(folder / "campaign") and unit["plan"] == str(folder / "plan.json"),
            "Unit paths differ",
        )
        plan = evidence.read(unit["plan"], unit["plan_sha256"], inside=folder)
        terminal = evidence.read(folder / "terminal_summary_v1.json")
        runner = evidence.read(folder / "runner_identity.json")
        require(runner["campaign"] == unit["campaign"], "Campaign runner identity differs")
        campaign = evidence.read(folder / "campaign/campaign.json", terminal["campaign_sha256"])
        require_terminal(campaign, runner_live=process_live(runner))
        require(campaign["status"] == terminal["status"], "Terminal status differs")
        require(campaign["last_audit"] == terminal["audit_id"], "Terminal audit differs")
        audit = evidence.read(folder / "campaign/audits" / (terminal["audit_id"] + ".json"), terminal["audit_sha256"])
        require(digest(audit) == terminal["audit_id"], "Audit content identity differs")
        plans[unit["id"]] = plan
        contexts[unit["id"]] = (folder, campaign, terminal)
    tasks = validate_design(design, plans)
    protocol = evidence.read(study / "final_protocol.json", design["final_protocol_sha256"])
    require(
        protocol["task_bindings"] == {tid: digest(task) for tid, task in tasks.items()},
        "Prospective task bindings differ",
    )
    require(
        protocol["fit_partition"] == "training" and protocol["score_partition"] == "future", "Scoring partitions differ"
    )
    declared = evidence.read(study / "tasks.json", design["tasks_sha256"])
    require(len(declared) == len(tasks) and {t["id"]: t for t in declared} == tasks, "Study tasks differ")
    return design, plans, contexts


def ranks(values):
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    first = 0
    while first < len(order):
        last = first + 1
        while last < len(order) and values[order[last]] == values[order[first]]:
            last += 1
        for i in order[first:last]:
            result[i] = (first + last + 1) / 2
        first = last
    return result


def replay_metrics(training, future, predictions, *, controls=False):
    """Recompute scalar metrics with tie-aware ranks, independent of evaluator."""
    require(
        len(predictions) == len(future) and [p["id"] for p in predictions] == [r["id"] for r in future],
        "Prediction identity/order differs",
    )
    values = [p["prediction"] for p in predictions]
    require(all(type(v) in (int, float) and math.isfinite(v) for v in values), "Nonfinite prediction")
    train = [r["target"] for r in training]
    y = [r["target"] for r in future]
    require(
        train and y and all(type(v) in (int, float) and math.isfinite(v) for v in train + y),
        "Finite labeled observations required",
    )
    mean = math.fsum(train) / len(train)
    mse = math.fsum((a - b) ** 2 for a, b in zip(y, values, strict=True)) / len(y)
    null = math.fsum((a - mean) ** 2 for a in y) / len(y)
    ry, rp = ranks(y), ranks(values)
    center = (len(y) + 1) / 2
    a, b = [v - center for v in ry], [v - center for v in rp]
    den = math.sqrt(math.fsum(v * v for v in a) * math.fsum(v * v for v in b))
    rho = math.fsum(x * z for x, z in zip(a, b, strict=True)) / den if den else None
    # The frozen control evaluator treats near-constant predictions as undefined.
    if controls and max(values) - min(values) <= 1e-12:
        rho = None
    return {
        "rmse": math.sqrt(mse),
        "mae": math.fsum(abs(x - z) for x, z in zip(y, values, strict=True)) / len(y),
        "spearman": rho,
        "mse_skill_vs_training_mean": 1 - mse / null if null else None,
    }


def equal_metrics(saved, replayed):
    for metric in METRICS:
        a, b = saved[metric], replayed[metric]
        require(
            a is None
            if b is None
            else type(a) in (int, float) and math.isfinite(a) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-10),
            "Metric replay differs: " + metric,
        )


def verify_task(evidence, task, row, prediction):
    require(
        row["task_sha256"] == digest(task) and row["scope"] == task["scope"] and row["data_bindings"] == task["data"],
        "Task report binding differs",
    )
    require(
        row["score_partition"] == "future" and row["fit_partition"] == "training" and row["future_scored"] is True,
        "Task score partitions differ",
    )
    data = {key: evidence.read(ref["path"], ref["sha256"]) for key, ref in task["data"].items()}
    for ref in task["references"]:
        evidence.read_bytes(ref["path"], ref["sha256"])
    require(set(data) == {"training", "validation", "future"}, "Three partitions required")
    for rows in data.values():
        require(len({r["id"] for r in rows}) == len(rows), "Duplicate observation identity")
    scientist = row["scientist"]
    if prediction is None:
        require(
            scientist == {"status": "not_registered", "metrics": None, "predictions": None},
            "Unpublished model received a score",
        )
    else:
        require(
            scientist["status"] == "scored" and scientist["predictions"] == prediction,
            "Score not derived from independent reuse",
        )
        expected = [(r["id"], r["protein"], r["group"], digest(r["values"])) for r in data["future"]]
        require(
            [(r.get("id"), r.get("protein"), r.get("group"), r.get("values_sha256")) for r in prediction] == expected,
            "Reuse prediction input witness differs",
        )
        equal_metrics(scientist["metrics"], replay_metrics(data["training"], data["future"], prediction))
    for control in row["controls"]:
        if control["status"] == "completed":
            equal_metrics(
                control["metrics"],
                replay_metrics(data["training"], data["future"], control["predictions"], controls=True),
            )
    # This refitter reconstructs all fixed estimators independently and checks groups.
    refit = verify_controls(task, row["controls"], partition="future")
    return refit


def unit_evidence(evidence, unit, plan, context, paths):
    folder, campaign, terminal = context
    reuse_path = evidence.path(paths["reuse"]["path"], inside=folder)
    eval_path = evidence.path(paths["evaluation"]["path"], inside=folder)
    require(reuse_path.name == eval_path.name == "report.json", "Explicit report.json paths required")
    reuse = evidence.read(reuse_path, paths["reuse"]["sha256"])
    scored = evidence.read(eval_path, paths["evaluation"]["sha256"])
    require(reuse["status"] == "independent_api_reuse_passed", "Independent API reuse has not passed")
    require(
        evidence.read(reuse_path.parent / "exit.json")["returncode"] == 0, "Independent reuse did not exit successfully"
    )
    rp = evidence.read(reuse_path.parent / "protocol.json", reuse["protocol_sha256"])
    predictions = evidence.read(reuse_path.parent / "inference_results.json", reuse["results_sha256"])
    receipt = evidence.read(reuse_path.parent / "independent_receipt_audit.json")
    require(
        receipt["status"] == "passed" and receipt["report_sha256"] == paths["reuse"]["sha256"],
        "Receipt audit missing or stale",
    )
    evidence.check_bindings(rp["source_bindings"])
    evidence.check_bindings(receipt["checked_files"])
    ep = evidence.read(eval_path.parent / "protocol.json", scored["protocol_sha256"])
    evidence.check_bindings(ep["source_bindings"])
    evidence.check_bindings(ep["implementation"])
    require(
        scored["status"] == "scored_pending_independent_replay" and scored["final_labels_scored"] is True,
        "Evaluation has not completed",
    )
    tasks = {t["id"]: t for t in plan["batches"][0]["tasks"]}
    require(
        ep["task_bindings"] == {tid: digest(task) for tid, task in tasks.items()}, "Evaluation task bindings differ"
    )
    require(
        len(ep["tasks"]) == len(tasks) and {t["id"]: t for t in ep["tasks"]} == tasks,
        "Evaluation task definitions differ",
    )
    require(set(scored["task_reports"]) == set(tasks), "Evaluation task coverage differs")
    published = {}
    for batch in campaign["batches"].values():
        for attempt in batch["attempts"]:
            for item in (attempt.get("publication") or {}).get("outcomes", []):
                require(
                    item["task_id"] in tasks and item["task_id"] not in published, "Duplicate or unassigned publication"
                )
                published[item["task_id"]] = item["entry_id"]
    rb = rp["task_bindings"]
    require(
        len(rb) == len(published) and {r["task_id"]: r["entry_id"] for r in rb} == published,
        "Reuse must cover all publications",
    )
    results = predictions["results"]
    require(len(set(published.values())) == len(published), "Entry reused across distinct tasks")
    indexed = {r["receipt"]["entry_id"]: r["predictions"] for r in results}
    require(
        len(results) == len(indexed) and set(indexed) == set(published.values()),
        "Duplicate or missing inference result",
    )
    for binding in rb:
        require(
            digest(indexed[binding["entry_id"]]) == binding["prediction_sha256"], "Independent prediction hash differs"
        )
    rows, refits = [], []
    for tid, task in tasks.items():
        row = evidence.read(eval_path.parent / (tid + ".json"), scored["task_reports"][tid])
        require(row["task_id"] == tid, "Task report ID differs")
        prediction = indexed[published[tid]] if tid in published else None
        refits.append(verify_task(evidence, task, row, prediction))
        rows.append(row)
    return rows, refits


def analyze(study, manifest, *, repository):
    evidence = Evidence(repository)
    study = evidence.path(study)
    require(
        manifest["schema"] == "matched_context_analysis_input.v1" and manifest["study"] == str(study),
        "Analysis manifest differs",
    )
    design, plans, contexts = terminal_study(evidence, study, manifest["launch_basis_sha256"])
    require(set(manifest["units"]) == set(plans), "Manifest must cover every unit")
    rows, refits, operations = {}, {}, {}
    for unit in design["units"]:
        uid = unit["id"]
        rows[uid], refits[uid] = unit_evidence(evidence, unit, plans[uid], contexts[uid], manifest["units"][uid])
        operations[uid] = collect_unit(evidence, plans[uid], contexts[uid])
        require(
            operations[uid]["registered"] == sum(r["scientist"]["status"] == "scored" for r in rows[uid]),
            "Operational and scored publication counts differ",
        )
    report = summarize(design, plans, rows)
    report["operational_analysis"] = summarize_operations(design, operations)
    evidence.unchanged()
    report.update(
        {
            "status": "matched_analysis_and_independent_metric_control_replay_passed",
            "source_bindings": evidence.bindings,
            "control_refits": refits,
            "final_labels_scored": True,
            "domain_baseline_comparison_complete": False,
        }
    )
    return report
