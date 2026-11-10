"""Deterministic task-disjoint support/query preparation from measured rows."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

from .contracts import digest
from .research_evaluation import EPISODE_SCHEMA


def prepare_episodes(rows, *, seed, support_size, validation_fraction, test_fraction, group_partitions=None):
    tasks = defaultdict(list)
    for row in rows:
        tasks[row["task_id"]].append(row)
    parents = {task: task for task in tasks}

    def root(task):
        while parents[task] != task:
            parents[task] = parents[parents[task]]
            task = parents[task]
        return task

    # A biological group shared by tasks connects those tasks. Keep the entire
    # component together rather than splitting a family across partitions.
    group_tasks = {}
    for row in rows:
        task = row["task_id"]
        prior = group_tasks.setdefault(row["group"], task)
        parents[root(task)] = root(prior)
    components = defaultdict(list)
    for task in sorted(tasks):
        components[root(task)].append(task)
    blocks = list(components.values())
    if len(blocks) < 3:
        raise ValueError("At least three independent task/group components are required")

    def order(value):
        return hashlib.sha256((str(seed)+"\0"+value).encode()).hexdigest(), value

    blocks.sort(key=lambda block: order(json.dumps(block)))
    if group_partitions is None:
        nv = max(1, round(len(blocks)*validation_fraction))
        nt = max(1, round(len(blocks)*test_fraction))
        if nv+nt >= len(blocks):
            raise ValueError("Holdout fractions leave no training tasks")
        assignment = {"validation": blocks[:nv], "test": blocks[nv:nv+nt], "train": blocks[nv+nt:]}
    else:
        assignment = {name: [] for name in ("validation", "test", "train")}
        if set(group_partitions) != {row["group"] for row in rows} or not set(group_partitions.values()) <= set(assignment):
            raise ValueError("Provide a valid partition for every biological group")
        for block in blocks:
            assigned = {group_partitions[row["group"]] for task in block for row in tasks[task]}
            if len(assigned) != 1:
                raise ValueError("A connected task/group component cannot cross provided partitions")
            assignment[assigned.pop()].append(block)
        if any(not selected for selected in assignment.values()):
            raise ValueError("Provided task groups must cover all three partitions")
    splits, supports = {}, {}
    for split, selected in assignment.items():
        queries, support_rows = [], []
        for block in selected:
            for task in block:
                ordered = sorted(tasks[task], key=lambda row: order("support\0"+task+"\0"+row["id"]))
                if len(ordered) <= support_size:
                    raise ValueError("Each task needs more observations than support_size; no task is silently discarded")
                support, query = ordered[:support_size], ordered[support_size:]
                support_inputs = {digest(row["inputs"]) for row in support}
                if len(support_inputs) != len(support) or any(digest(row["inputs"]) in support_inputs for row in query):
                    raise ValueError("Support and query observations must have distinct inputs; consolidate repeated measurements first")
                observations = [{"id": row["id"], "inputs": row["inputs"], "target": row["target"]} for row in support]
                queries.extend({**row, "inputs": {**row["inputs"], "support": observations}} for row in query)
                support_rows.extend(support)
        splits[split] = queries
        supports[split] = support_rows
    return splits, supports, {"schema": EPISODE_SCHEMA, "support_size": support_size,
                              "support_key": "support", "split_method": ("provided_connected_task_groups" if group_partitions is not None
                                               else "whole_connected_task_groups")}
