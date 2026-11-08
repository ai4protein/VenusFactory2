"""Request-local batch inference with complete provenance checks before and after.

Only terminal shared-goal verification is shared within one call. No validation
result survives the request; legacy and migrated models use the existing resolver.
"""

from __future__ import annotations

import copy
import time
import uuid

from agent.harness.contracts import digest
from agent.research_modes import get_research_mode

from agent.rsi import model_tasks
from agent.rsi.engine.model_plugins import get_model_plugin
from agent.rsi.engine.scientist_model_library import _require
from agent.rsi.engine.storage import file_lock

MAX_BATCH_REQUESTS = 256


def _resolve_all(library, entry_ids):
    resolved = library.resolve_many(entry_ids)
    shared_goals = {
        value["origin_goal"]
        for value in resolved.values()
        if value["origin"].get("evidence_kind") == "native_scientist_shared_program_model"
    }
    return resolved, len(shared_goals)


def predict_many(library, requests):
    """Infer a bounded batch; every request has its own distinct session and receipt."""
    _require(
        isinstance(requests, list) and 1 <= len(requests) <= MAX_BATCH_REQUESTS, "Declare 1..256 inference requests"
    )
    requests = copy.deepcopy(requests)
    sessions = set()
    for request in requests:
        _require(
            isinstance(request, dict)
            and set(request)
            in (
                {"entry_id", "inputs", "scope", "session_id"},
                {"entry_id", "inputs", "scope", "session_id", "mode"},
            ),
            "Invalid batch inference request fields",
        )
        session = request["session_id"]
        _require(isinstance(session, str) and bool(session.strip()), "Inference requires a session identifier")
        _require(session not in sessions, "Batch inference sessions must be distinct")
        sessions.add(session)
        get_research_mode(request.get("mode", "science_agent"))
    started = time.monotonic()
    verified, shared_goals = _resolve_all(library, [r["entry_id"] for r in requests])
    checked = time.monotonic()
    batch_id = uuid.uuid4().hex
    results = []
    for request in requests:
        entry_id, scope = request["entry_id"], request["scope"]
        model = verified[entry_id]["packet"]["model"]
        _require(scope == model["scope"], "Model applicability mismatch")
        rows = model_tasks.inference_inputs(request["inputs"], model["plugin_id"])
        predictions = get_model_plugin(model["plugin_id"]).predict(model, rows, scope)
        receipt = {
            "schema": 1,
            "kind": "model_library_inference",
            "call_id": uuid.uuid4().hex,
            "batch_id": batch_id,
            "entry_id": entry_id,
            "session_id": request["session_id"],
            "mode": request.get("mode", "science_agent"),
            "scope_sha256": digest(scope),
            "input_sha256": digest(rows),
            "prediction_sha256": digest(predictions),
            "rows": len(rows),
        }
        results.append({"predictions": predictions, "receipt": receipt})
    predicted = time.monotonic()
    after, after_goals = _resolve_all(library, [r["entry_id"] for r in requests])
    _require(after == verified and after_goals == shared_goals, "Model evidence changed during batch inference")
    rechecked = time.monotonic()
    # No receipt is written until every inference and the complete second audit
    # have passed. Recheck registry bindings while holding the writer lock.
    with file_lock(library.root / ".library.lock"):
        for entry_id, value in verified.items():
            library._bind_verified(entry_id, library._binding(entry_id), value)
        for result in results:
            result["receipt_id"] = library._immutable("usage", result["receipt"])
        batch = {
            "schema": "model_library_batch_inference.v1",
            "batch_id": batch_id,
            "requests": len(requests),
            "unique_entries": len(verified),
            "verified_shared_goals": shared_goals,
            "verification_passes": 2,
            "persistent_validation_cache": False,
            "receipt_ids": [r["receipt_id"] for r in results],
            "verification_sha256": digest(verified),
            "rows": sum(r["receipt"]["rows"] for r in results),
            "seconds": {
                "initial_verification": checked - started,
                "prediction": predicted - checked,
                "final_verification": rechecked - predicted,
            },
            "external_model_calls": 0,
            "new_autonomous_models": 0,
        }
        batch_receipt_id = library._immutable("usage_batches", batch)
    return {"results": results, "batch_receipt": batch, "batch_receipt_id": batch_receipt_id}
