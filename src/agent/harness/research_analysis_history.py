"""Revalidate prior host-captured analyses without rewriting episode receipts.

The caller supplies cycles from GoalStore, never workspace/model-authored state.
This is a private host view; only its verified projection belongs in SC context.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from utils.scoped_files import read_bytes

from .research_analysis_evidence import JOB, VerificationBudget
from .research_analysis_provenance import _verified_document

SCHEMA = "research-analysis-history.v1"
MAX_CYCLES = 128
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_ROWS = 256
MAX_ISSUES = 64


def _issue(issues, code, **fields):
    if len(issues) < MAX_ISSUES:
        issues.append({"code": code, **fields})
    elif len(issues) == MAX_ISSUES:
        issues.append({"code": "additional_history_issues_omitted"})


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate private receipt field")
        result[key] = value
    return result


def _invalid_constant(_value):
    raise ValueError("Non-finite private receipt value")


def load_history(goal_root, *, workspace, cycles):
    """Read only known completed/failed episode controls inside this goal.

    Entries that cannot be verified later are diagnostics, not global blockers.
    Raw private records are retained here solely to detect conflicting task IDs
    even when one of the conflicting variants is invalid.
    """
    root, workspace = Path(goal_root), Path(workspace)
    if root.is_symlink() or workspace.is_symlink() or workspace.resolve() != (root / "workspace").resolve():
        raise ValueError("Analysis history must belong to this goal workspace")
    indices = []
    issues = []
    for row in cycles:
        index = row.get("index") if isinstance(row, dict) else None
        if type(index) is not int or not 0 <= index <= 999999:
            _issue(issues, "invalid_cycle_identity")
            continue
        if index not in indices:
            indices.append(index)
    indices.sort()
    if len(indices) > MAX_CYCLES:
        _issue(issues, "cycle_limit", omitted=len(indices) - MAX_CYCLES)
        indices = indices[-MAX_CYCLES:]
    records, sources, origins, remaining = [], [], {}, MAX_TOTAL_BYTES
    for index in indices:
        relative = f"episodes/cycle_{index:03d}/tools/receipts.json"
        if remaining <= 0 or len(records) >= MAX_ROWS:
            _issue(issues, "history_limit", cycle_index=index)
            break
        try:
            raw = read_bytes(root, relative, max_bytes=min(MAX_FILE_BYTES, remaining))
            remaining -= len(raw)
            rows = json.loads(raw, object_pairs_hook=_unique, parse_constant=_invalid_constant)
            if not isinstance(rows, list):
                raise ValueError("Private receipts must be a list")
            sources.append({"cycle_index": index, "sha256": hashlib.sha256(raw).hexdigest()})
            for row in rows:
                if not isinstance(row, dict) or row.get("tool") != "queued_analysis":
                    continue
                if len(records) >= MAX_ROWS:
                    _issue(issues, "receipt_limit", cycle_index=index)
                    break
                records.append(copy.deepcopy(row))
                saved = row.get("result")
                identity = saved.get("task_id") if isinstance(saved, dict) else None
                if isinstance(identity, str) and JOB.fullmatch(identity):
                    origins.setdefault(identity, []).append({
                        "cycle_index": index, "receipt_file_sha256": hashlib.sha256(raw).hexdigest()})
        except FileNotFoundError:
            # An episode may have failed before any scientific tool was invoked.
            continue
        except (OSError, ValueError, TypeError, RecursionError):
            _issue(issues, "private_receipts_unavailable_or_invalid", cycle_index=index)
    return {"schema": SCHEMA, "records": records, "sources": sources, "origins": origins, "issues": issues}


def merge_receipts(history, current_receipts, workspace, *, checkpoint=None):
    """Return a revalidated view; never append old rows to current run receipts."""
    if not isinstance(history, dict) or history.get("schema") != SCHEMA:
        raise ValueError("Unsupported analysis history")
    issues = copy.deepcopy(history.get("issues", []))
    candidates = [*history.get("records", []),
                  *(row for row in current_receipts if row.get("tool") == "queued_analysis")]
    rows, fingerprints, conflicted = {}, {}, set()
    for row in candidates:
        saved = row.get("result", {})
        identity = saved.get("task_id") if isinstance(saved, dict) else None
        if not isinstance(identity, str) or not JOB.fullmatch(identity):
            _issue(issues, "invalid_private_task_identity")
            continue
        try:
            canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        except (ValueError, TypeError, RecursionError):
            conflicted.add(identity)
            _issue(issues, "invalid_private_receipt", task_id=identity)
            continue
        fingerprint = hashlib.sha256(canonical).hexdigest()
        if identity in fingerprints and fingerprints[identity] != fingerprint:
            conflicted.add(identity)
            _issue(issues, "conflicting_private_task", task_id=identity)
        else:
            fingerprints[identity] = fingerprint
            rows.setdefault(identity, row)
    current_ids = {row.get("result", {}).get("task_id") for row in current_receipts
                   if row.get("tool") == "queued_analysis" and isinstance(row.get("result"), dict)}
    cited = {row.get("path") for row in (checkpoint or {}).get("artifacts", [])}
    def priority(item):
        identity, row = item
        return (0 if identity in current_ids else
                1 if row.get("result", {}).get("artifact", {}).get("path") in cited else 2)
    verified_ids = set()
    budget = VerificationBudget()
    for identity, row in sorted(rows.items(), key=priority):
        if identity in conflicted:
            continue
        try:
            _verified_document(workspace, row, budget=budget)
        except (OSError, ValueError, TypeError, KeyError, RecursionError):
            _issue(issues, "historical_or_current_artifact_unverified", task_id=identity)
            continue
        verified_ids.add(identity)
    # Keep original history-to-current ordering after prioritized bounded verification.
    verified = [copy.deepcopy(row) for identity, row in rows.items() if identity in verified_ids]
    provenance = {identity: {
        "scope": "current" if identity in current_ids else "prior_episode",
        "prior_capture_sources": copy.deepcopy(history.get("origins", {}).get(identity, [])),
    } for identity in verified_ids}
    # Current non-analysis receipts still drive their existing validators. They
    # are not added to the analysis projection and are not imported from history.
    current = [copy.deepcopy(row) for row in current_receipts if row.get("tool") != "queued_analysis"]
    return {"receipts": [*current, *verified], "issues": issues, "task_provenance": provenance,
            "history_sources": copy.deepcopy(history.get("sources", []))}
