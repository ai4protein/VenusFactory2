"""Read-only acquisition evidence shared by individual and batch evaluations."""

import json
import math
from pathlib import Path

from agent.harness.contracts import file_hash

from .source_snapshot import research_implementation_paths

REPOSITORY = Path(__file__).resolve().parents[4]


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    require(path.is_file() and not path.is_symlink(), "Missing or symlinked evaluation evidence")
    return json.loads(path.read_text())


def source_identity():
    return {str(p.relative_to(REPOSITORY)): file_hash(p) for p in research_implementation_paths(REPOSITORY)}


def tree(root):
    if not root.exists():
        return {}
    require(root.is_dir() and not root.is_symlink(), "Invalid goal evidence directory")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "Symlinked goal evidence")
        if path.is_file() and not path.name.endswith(".lock"):
            result[str(path.relative_to(root))] = file_hash(path)
    return result


def observed_usage(root, record):
    physical, tools, token_rows, host_seconds, missing = 0, 0, [], 0.0, []
    for folder in sorted((root / "episodes").glob("cycle_*/execution")):
        request_file = folder / "model_requests.jsonl"
        events_file = folder / "events.jsonl"
        if not request_file.exists() or not events_file.exists():
            missing.append(str(folder.relative_to(root)))
        requests = [json.loads(line) for line in request_file.read_text().splitlines()] if request_file.exists() else []
        ids = [row["call"] for row in requests if row.get("event") == "request"]
        require(len(ids) == len(set(ids)), "Duplicate physical request record")
        physical += len(ids)
        responses = [row for row in requests if row.get("event") == "response"]
        require(len({row["call"] for row in responses}) == len(responses), "Duplicate response usage")
        for row in responses:
            require(row["call"] in ids, "Unpaired response usage")
            usage = row.get("usage") or {}
            values = [
                usage.get("prompt_tokens", usage.get("input_tokens")),
                usage.get("completion_tokens", usage.get("output_tokens")),
            ]
            valid = all(type(v) is int and v >= 0 for v in values)
            token_rows.append(values if valid else None)
        token_rows.extend([None] * (len(ids) - len(responses)))
        if events_file.exists():
            tools += sum(
                json.loads(line).get("kind") == "tool_call_start" for line in events_file.read_text().splitlines()
            )
        run_file = folder / "run.json"
        if run_file.exists():
            seconds = read(run_file).get("host_wall_seconds")
            if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
                host_seconds += seconds
            else:
                missing.append(str(run_file.relative_to(root)))
        else:
            missing.append(str(run_file.relative_to(root)))
    known = bool(record["attempts"]) and all(a.get("usage_known") is True for a in record["attempts"])
    charged = {
        k: sum(a["charged"][k] for a in record["attempts"]) for k in ("model_calls", "tool_calls", "wall_seconds")
    }
    complete = known and not missing and charged["model_calls"] == physical and charged["tool_calls"] == tools
    return {
        "physical_requests_observed": physical,
        "tool_calls_observed": tools,
        "call_accounting_complete": complete,
        "input_tokens_observed": sum(v[0] for v in token_rows if v is not None),
        "output_tokens_observed": sum(v[1] for v in token_rows if v is not None),
        "token_accounting_complete": complete and all(v is not None for v in token_rows),
        "host_execution_seconds_observed": host_seconds,
        "charged_budget": charged,
        "missing_execution_evidence": missing,
        "scope": "Observed values are lower bounds when accounting is incomplete; charged reservations are reported separately",
    }
