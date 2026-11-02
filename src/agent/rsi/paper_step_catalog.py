"""Host-configured paper computation definitions; clients select stable IDs only."""

import json
import re
from pathlib import Path

from .paper_step import validate_definition
from .storage import safe_child, sha256_file


def catalog(root):
    root = Path(root).resolve()
    manifest = safe_child(root, "configs/rsi/paper_steps.json")
    if not manifest.exists():
        return {}
    if manifest.stat().st_size > 131072:
        raise ValueError("Paper step catalog is too large")
    value = json.loads(manifest.read_text())
    if (
        set(value) != {"schema", "tasks"}
        or value["schema"] != 1
        or not isinstance(value["tasks"], dict)
        or len(value["tasks"]) > 100
    ):
        raise ValueError("Invalid paper step catalog")
    result = {}
    for identity, row in value["tasks"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", identity) or set(row) != {"label", "definition", "sha256"}:
            raise ValueError("Invalid catalog task identity or fields")
        if set(row["label"]) != {"en", "zh"} or any(
            not isinstance(x, str) or not 1 <= len(x) <= 200 for x in row["label"].values()
        ):
            raise ValueError("Task labels are required")
        allowed_roots = ("research_runs/case_definitions/", "configs/rsi/paper_step_definitions/")
        prefix = next((item for item in allowed_roots if isinstance(row["definition"], str)
                       and row["definition"].startswith(item)), None)
        if prefix is None:
            raise ValueError("Task definitions must be host catalog files")
        path = safe_child(root, row["definition"])
        path.relative_to(root / prefix)
        if path.suffix != ".json" or path.stat().st_size > 64 * 1024**2 or sha256_file(path) != row["sha256"]:
            raise ValueError("Catalog definition changed or exceeds its size bound")
        definition = validate_definition(json.loads(path.read_text()))
        result[identity] = {
            "id": identity,
            "label": row["label"],
            "definition": definition,
            "definition_sha256": row["sha256"],
        }
    return result


def task(root, identity):
    tasks = catalog(root)
    if not isinstance(identity, str) or identity not in tasks:
        raise ValueError("Unknown host-admitted computation task")
    return tasks[identity]
