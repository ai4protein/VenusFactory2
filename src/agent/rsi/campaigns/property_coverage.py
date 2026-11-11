"""Allocate scientific properties before proteins, using declarations only."""

from __future__ import annotations

from collections import Counter, defaultdict, deque

from agent.harness.contracts import digest
from agent.rsi.datasets.research_project_catalog import identifier, require, text

SCHEMA = "property_coverage_policy.v1"


def validate_policy(policy):
    require(
        isinstance(policy, dict) and set(policy) == {"schema", "families", "rules", "max_per_protein"},
        "Invalid property coverage policy fields",
    )
    require(policy["schema"] == SCHEMA, "Unsupported property coverage policy")
    require(
        type(policy["max_per_protein"]) is int and policy["max_per_protein"] > 0,
        "Positive per-protein allocation cap required",
    )
    families = policy["families"]
    require(isinstance(families, dict) and families, "Property families required")
    for name, limits in families.items():
        identifier(name)
        require(
            isinstance(limits, dict) and set(limits) == {"target", "max_per_study", "min_studies"},
            "Invalid family allocation limits",
        )
        require(all(type(v) is int and v >= 0 for v in limits.values()), "Nonnegative integer limits required")
        if limits["target"]:
            require(
                1 <= limits["max_per_study"] <= limits["target"] and 1 <= limits["min_studies"] <= limits["target"],
                "Inconsistent family limits",
            )
        else:
            require(not any(limits.values()), "A deferred family must have zero limits")
    require(sum(v["target"] for v in families.values()) > 0, "At least one active family required")
    require(isinstance(policy["rules"], list) and policy["rules"], "Explicit property classification rules required")
    seen, wildcards = set(), set()
    for rule in policy["rules"]:
        require(
            isinstance(rule, dict) and {"stage", "family"} <= set(rule) <= {"stage", "family", "targets"},
            "Invalid property rule",
        )
        stage = text(rule["stage"])
        require(rule["family"] in families, "Unknown property family")
        if "targets" not in rule:
            require(stage not in wildcards and not any(s == stage for s, _ in seen), "Overlapping property rules")
            wildcards.add(stage)
        else:
            require(isinstance(rule["targets"], list) and rule["targets"], "Nonempty target labels required")
            for target in rule["targets"]:
                key = (stage, text(target))
                require(stage not in wildcards and key not in seen, "Overlapping property rules")
                seen.add(key)
    return sum(v["target"] for v in families.values())


def classify(policy, stage, target):
    stage, target = text(stage), text(target)
    for rule in policy["rules"]:
        if text(rule["stage"]) == stage and ("targets" not in rule or target in {text(t) for t in rule["targets"]}):
            return rule["family"]
    return None


def study_identity(value):
    value = text(value)
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
            break
    require(bool(value), "Study identity required")
    return value


def select_by_property(records, policy):
    """Apply fixed family/study/protein caps; never backfill a missing family.

    Records contain source declarations, not predictions or acceptance scores.
    Project aliases sharing a DOI consume the same study quota. Scenario and
    input-interface coverage are reported independently of protein identities.
    """
    requested = validate_policy(policy)
    queues, metadata, skipped = defaultdict(deque), {}, Counter()
    available = Counter()
    for record in sorted(records, key=lambda r: r["id"]):
        name = record["id"]
        require(isinstance(name, str) and name and name not in metadata, "Distinct task identities required")
        metadata[name] = record
        family = classify(policy, record["stage"], record["target"])
        if family is None:
            skipped["unclassified_property"] += 1
            continue
        available[family] += 1
        if not policy["families"][family]["target"]:
            skipped["deferred_property_family"] += 1
            continue
        study = study_identity(record["study_doi"])
        scenario = record["scenario"]
        require(isinstance(scenario, dict) and scenario, "Measured scenario required")
        scenario = {text(k): text(v) for k, v in scenario.items()}
        scenario_key = digest(
            {"scenario": scenario, "target": text(record["target"]), "interface": text(record["interface"])}
        )
        queues[family, study, scenario_key].append(name)
    chosen, family_counts, study_counts, scenario_counts, protein_counts = (
        [],
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    assignments, interfaces, selected_studies = {}, defaultdict(set), defaultdict(set)
    while True:
        progress = False
        for family, limits in sorted(policy["families"].items()):
            if family_counts[family] >= limits["target"]:
                continue
            options = []
            for key, queue in queues.items():
                f, study, scenario = key
                if f != family or study_counts[f, study] >= limits["max_per_study"]:
                    continue
                while queue and protein_counts[text(metadata[queue[0]]["protein"])] >= policy["max_per_protein"]:
                    queue.popleft()
                    skipped["per_protein_cap"] += 1
                if queue:
                    options.append(key)
            if not options:
                continue
            key = min(options, key=lambda k: (study_counts[k[0], k[1]], scenario_counts[k[0], k[2]], k))
            name = queues[key].popleft()
            _, study, scenario = key
            chosen.append(name)
            assignments[name] = family
            family_counts[family] += 1
            study_counts[family, study] += 1
            scenario_counts[family, scenario] += 1
            protein_counts[text(metadata[name]["protein"])] += 1
            selected_studies[family].add(study)
            interfaces[family].add(metadata[name]["interface"])
            progress = True
        if not progress:
            break
    rows = {}
    for family, limits in sorted(policy["families"].items()):
        rows[family] = {
            **limits,
            "available_candidates": available[family],
            "selected": family_counts[family],
            "shortfall": limits["target"] - family_counts[family],
            "studies": sorted(selected_studies[family]),
            "minimum_studies_met": len(selected_studies[family]) >= limits["min_studies"],
            "scenarios": sum(1 for f, _ in scenario_counts if f == family),
            "interfaces": sorted(interfaces[family]),
        }
    return {
        "schema": SCHEMA,
        "policy_sha256": digest(policy),
        "requested_tasks": requested,
        "selected_tasks": len(chosen),
        "selected_ids": chosen,
        "assignments": assignments,
        "families": rows,
        "skipped": dict(sorted(skipped.items())),
        "coverage_satisfied": all(not v["shortfall"] and v["minimum_studies_met"] for v in rows.values()),
        "distinct_protein_labels": len(protein_counts),
        "selection_basis": "Property quotas, study rotation and measured scenarios; no model outcomes.",
        "new_autonomous_models": 0,
    }
