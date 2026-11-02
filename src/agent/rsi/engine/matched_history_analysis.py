"""Paired, source-stratified summaries for frozen three-arm context studies."""

import math
from collections import defaultdict
from itertools import permutations

import numpy as np

from .protein_model import digest

ARMS = ("no_history", "recent", "selective")
CONTRASTS = (("selective", "recent"), ("recent", "no_history"), ("selective", "no_history"))
METRICS = ("mse_skill_vs_training_mean", "spearman", "rmse", "mae")
SEED = 20261027
RESAMPLES = 10000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_design(design, plans):
    """Validate allocation metadata without reading measurement partitions."""
    units = design["units"]
    require(len({u["id"] for u in units}) == len(units), "Duplicate unit")
    require(set(plans) == {u["id"] for u in units}, "Plan coverage differs")
    require(set(design["condition_policies"]) == set(ARMS), "Three declared history policies required")
    blocks, orders, tasks, scopes = {}, defaultdict(list), {}, set()
    for unit in units:
        require(unit["arm"] in ARMS, "Unknown arm")
        block = (unit["project"], unit["source_block"])
        record = blocks.setdefault(block, {})
        require(unit["arm"] not in record, "Repeated arm in block")
        plan = plans[unit["id"]]
        require(len(plan["batches"]) == 1, "Exactly one shared batch per unit required")
        batch = plan["batches"][0]
        require(batch["shared_training_history"] == design["condition_policies"][unit["arm"]], "History policy differs")
        require(batch["profile"]["model"] == design["free_model_only"], "Model profile differs")
        require(batch["goal_budget"] == design["budget_per_unit"], "Unit budget differs")
        require(
            plan["resources"] == {k: v for k, v in design["budget_per_unit"].items() if k != "max_cycles"},
            "Resource budget differs",
        )
        require(len(batch["tasks"]) == plan["target_scopes"] == design["tasks_per_block"], "Task count differs")
        assigned = {t["id"]: t for t in batch["tasks"]}
        require(len(assigned) == len(batch["tasks"]), "Repeated task in unit")
        require(
            all(
                t["plugin_id"] in {"generated_context.v1", "generated_context.v2", "generated_context.v3"}
                for t in assigned.values()
            ),
            "Context tasks required",
        )
        # Preserve every factor other than the explicitly assigned history policy.
        normalized = {**plan, "batches": [{k: v for k, v in batch.items() if k != "shared_training_history"}]}
        record[unit["arm"]] = (unit["id"], digest(normalized), assigned)
    require(len(blocks) == design["matched_blocks"], "Block count differs")
    for (project, _block), record in blocks.items():
        require(set(record) == set(ARMS), "Incomplete three-arm block")
        require(len({value[1] for value in record.values()}) == 1, "Non-policy factors differ within block")
        orders[project].append(tuple(record))
        assigned = record[ARMS[0]][2]
        require(not set(tasks) & set(assigned), "Task repeated across blocks")
        for task in assigned.values():
            scope = digest(task["scope"])
            require(scope not in scopes, "Scientific scope repeated across blocks")
            scopes.add(scope)
        tasks.update(assigned)
    require(set(orders) == set(design["source_projects"]), "Source coverage differs")
    for values in orders.values():
        require(len(values) == 6 and set(values) == set(permutations(ARMS)), "Six arm permutations required per source")
    require(len(tasks) == len(scopes) == design["scientific_scopes"], "Scientific scope count differs")
    require(len(tasks) * len(ARMS) == design["allocated_model_attempts"], "Allocation count differs")
    return tasks


def interval(block_values):
    """Paired block bootstrap, resampled within each source; never per target."""
    groups = defaultdict(list)
    for project, _block, value in sorted(block_values):
        require(type(value) in (int, float) and math.isfinite(value), "Finite block effect required")
        groups[project].append(float(value))
    n = sum(map(len, groups.values()))
    if not n:
        return {"estimate": None, "interval_95": None, "blocks": 0, "source_blocks": {}, "degenerate": None}
    estimate = sum(sum(values) for values in groups.values()) / n
    # A single independent block cannot estimate between-block uncertainty.
    ci = None
    if all(len(values) >= 2 for values in groups.values()):
        rng = np.random.default_rng(SEED)
        samples = np.zeros(RESAMPLES)
        for values in groups.values():
            a = np.asarray(values)
            samples += a[rng.integers(0, len(a), size=(RESAMPLES, len(a)))].sum(axis=1)
        ci = [float(v) for v in np.quantile(samples / n, [0.025, 0.975])]
    return {
        "estimate": estimate,
        "interval_95": ci,
        "blocks": n,
        "source_blocks": {key: len(value) for key, value in groups.items()},
        "degenerate": ci[0] == ci[1] if ci is not None else None,
        "uncertainty": "Descriptive source-stratified paired-block percentile bootstrap; a zero-width interval does not establish equivalence.",
    }


def summarize(design, plans, unit_rows):
    """Consume independently verified per-task rows; keep all assigned scopes."""
    tasks = validate_design(design, plans)
    units = {u["id"]: u for u in design["units"]}
    require(set(unit_rows) == set(units), "Every allocated unit must be included")
    blocks, arms = {}, {arm: [] for arm in ARMS}
    for uid, unit in units.items():
        rows = unit_rows[uid]
        expected = {t["id"]: t for t in plans[uid]["batches"][0]["tasks"]}
        require(
            len(rows) == len(expected) and {r["task_id"] for r in rows} == set(expected), "Task report coverage differs"
        )
        indexed = {}
        for row in rows:
            require(row["task_sha256"] == digest(expected[row["task_id"]]), "Task report identity differs")
            scientist = row["scientist"]
            require(scientist["status"] in {"scored", "not_registered"}, "Unknown acquisition status")
            if scientist["status"] == "not_registered":
                require(
                    scientist["metrics"] is None and scientist["predictions"] is None,
                    "Unregistered model contains scores",
                )
            else:
                for metric in METRICS:
                    value = scientist["metrics"][metric]
                    require(value is None or (type(value) in (int, float) and math.isfinite(value)), "Nonfinite metric")
                    require(
                        value is not None or metric in {"spearman", "mse_skill_vs_training_mean"},
                        "Error metric missing",
                    )
            indexed[row["task_id"]] = row
        blocks.setdefault((unit["project"], unit["source_block"]), {})[unit["arm"]] = indexed
        arms[unit["arm"]].extend(rows)
    contrasts = {}
    for left, right in CONTRASTS:
        acquisition, metric_values = [], defaultdict(list)
        paired_counts = defaultdict(int)
        block_rows = []
        for (project, block), values in sorted(blocks.items()):
            a, b = values[left], values[right]
            acquired_a = sum(r["scientist"]["status"] == "scored" for r in a.values())
            acquired_b = sum(r["scientist"]["status"] == "scored" for r in b.values())
            effect = (acquired_a - acquired_b) / len(a)
            acquisition.append((project, block, effect))
            block_rows.append(
                {
                    "project": project,
                    "block": block,
                    "allocated_per_arm": len(a),
                    "left_registered": acquired_a,
                    "right_registered": acquired_b,
                    "acquisition_difference": effect,
                }
            )
            for metric in METRICS:
                differences = []
                for task_id in sorted(a):
                    ma, mb = a[task_id]["scientist"]["metrics"], b[task_id]["scientist"]["metrics"]
                    if ma is not None and mb is not None and ma[metric] is not None and mb[metric] is not None:
                        differences.append(ma[metric] - mb[metric])
                if differences:
                    value = sum(differences) / len(differences)
                    metric_values[(project, metric)].append((project, block, value))
                    paired_counts[(project, metric)] += len(differences)
        quality = {}
        for metric in METRICS:
            source_results = {
                project: {
                    **interval(metric_values[(project, metric)]),
                    "paired_tasks": paired_counts[(project, metric)],
                }
                for project in sorted(design["source_projects"])
            }
            item = {
                "direction": "positive_favors_left" if metric in METRICS[:2] else "negative_favors_left",
                "by_source": source_results,
            }
            if metric in METRICS[:2]:
                item["pooled"] = {
                    **interval([v for (p, m), values in metric_values.items() if m == metric for v in values]),
                    "paired_tasks": sum(count for (p, m), count in paired_counts.items() if m == metric),
                }
            quality[metric] = item
        contrasts[left + "_minus_" + right] = {
            "left": left,
            "right": right,
            "primary": (left, right) == CONTRASTS[0],
            "acquisition": interval(acquisition),
            "block_effects": block_rows,
            "conditional_paired_quality": quality,
        }
    return {
        "schema": "matched_context_history_analysis.v1",
        "scientific_scopes": len(tasks),
        "allocated_model_attempts": sum(map(len, unit_rows.values())),
        "distinct_registered_scopes": len(
            {r["task_id"] for rows in unit_rows.values() for r in rows if r["scientist"]["status"] == "scored"}
        ),
        "arms": {
            arm: {
                "allocated": len(rows),
                "registered_and_independently_reused": sum(r["scientist"]["status"] == "scored" for r in rows),
                "not_registered": sum(r["scientist"]["status"] == "not_registered" for r in rows),
            }
            for arm, rows in arms.items()
        },
        "contrasts": contrasts,
        "seed": SEED,
        "resamples": RESAMPLES,
        "analysis_unit": "shared-program block within source study",
        "quality_population": "Both arms acquired the task and both metrics are defined; missing values are not imputed. Equal weight per contributing block, then proportional weight by source block count.",
        "statistical_independence_assumed_between_targets": False,
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
