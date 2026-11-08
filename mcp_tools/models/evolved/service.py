"""Small local Agent interface to host-verified Scientist model publications."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from agent.rsi.engine.scientist_model_library import ScientistModelLibrary, _require
from mcp_tools.tools.runtime_tool_policy import assert_tool_allowed

TOOL_NAMES = {
    "list": "list_evolved_models",
    "describe": "describe_evolved_model",
    "predict": "predict_with_evolved_model",
}


def configured_library():
    """The deployment operator owns this path; it is never a model tool argument."""
    from mcp_tools.common.installation import scientist_library_root
    return ScientistModelLibrary(scientist_library_root(Path(__file__).resolve().parents[3]))


def public_card(entry_id, key, origin, *, verified_now):
    return {
        "entry_id": entry_id,
        "interface": key["plugin_id"],
        "scope": key["scope"],
        "stage": origin.get("stage", "unspecified"),
        "paper": origin.get("paper"),
        "evidence_kind": origin.get("evidence_kind"),
        "runtime_migrated": bool(origin.get("runtime_migration")),
        "verified_now": verified_now,
    }


def discovery_card(card):
    """Bounded discovery metadata; exact inputs and evidence live in describe."""
    scope = card["scope"]
    protein = str(scope.get("protein", "Protein model"))[:80]
    target = str(scope.get("target", "prediction")).replace("_", " ")[:100]
    assay = str(scope.get("assay", "")).replace("_", " ")[:120]
    return {
        "entry_id": card["entry_id"],
        "name": f"{protein} · {target}"[:160],
        "description": f"Predict {target}. {assay}".strip()[:240],
        "stage": card["stage"],
        "interface": card["interface"],
        "protein": protein,
        "target": target,
        "verified_now": False,
        "describe": {"tool": "describe_evolved_model", "arguments": {"entry_id": card["entry_id"]}},
        "predict_tool": "predict_with_evolved_model",
    }


def catalog_cards(library):
    """Public publication metadata, shared by listing and capability search."""
    from agent.harness.contracts import digest

    catalog = library.catalog()
    cards = []
    implementations = {}
    from agent.rsi.engine.model_plugins import get_model_plugin
    for item in catalog["entries"]:
        entry = library._object("entries", item["entry_id"])
        origin = library._object("origins/" + digest(item["key"]), entry["canonical_origin"])
        plugin_id = item["key"]["plugin_id"]
        if plugin_id not in implementations:
            try:
                implementations[plugin_id] = get_model_plugin(plugin_id).source_sha256
            except (ValueError, KeyError, FileNotFoundError):
                implementations[plugin_id] = None
        packet = library._object("packets", entry["packet_id"])
        compatible = bool(implementations[plugin_id]) and packet["assessment"].get("plugin_sha256") == implementations[plugin_id]
        cards.append({**public_card(item["entry_id"], item["key"], origin, verified_now=False),
                      "runtime_compatible": compatible})
    return cards, catalog


def list_models(library, query="", stage="", limit=10, offset=0):
    _require(type(limit) is int and 1 <= limit <= 50, "Limit must be between 1 and 50")
    _require(type(offset) is int and offset >= 0, "Invalid catalog offset")
    _require(
        isinstance(query, str) and len(query) <= 200 and isinstance(stage, str) and len(stage) <= 80,
        "Invalid catalog filter",
    )
    cards = []
    published, catalog = catalog_cards(library)
    from mcp_tools.tools.capability_policy import CapabilityStore
    switches = CapabilityStore()
    for card in published:
        if not card.get("runtime_compatible", True): continue
        if not switches.enabled('model:scientist:' + card['entry_id']): continue
        if stage and card["stage"] != stage:
            continue
        if query.casefold() not in json.dumps(card, ensure_ascii=False).casefold():
            continue
        cards.append(card)
    return {
        "models": [discovery_card(card) for card in cards[offset : offset + limit]],
        "matching_entries": len(cards),
        "indexed_entries": catalog["indexed_entries"],
        "next_offset": offset + limit if offset + limit < len(cards) else None,
        "availability": "Recorded publications; describe and predict revalidate current evidence.",
        "counting_rule": catalog["counting_rule"],
    }


def describe_model(library, entry_id):
    verified = library.resolve(entry_id)
    key = verified["key"]
    from agent.rsi.engine.context_interfaces import PLUGINS

    contextual = key["plugin_id"] in PLUGINS
    if key["plugin_id"] == "generated_antibody.v1":
        from agent.rsi.engine.generated_antibody import INPUT_FIELDS

        fields = list(INPUT_FIELDS)
        contract = {
            "kind": "paired_antibody_measurement",
            "representation": key["scope"]["representation"],
            "chains": "Separate variable heavy and light sequences; light is empty only for declared VHH tasks.",
            "values": {
                "type": "ordered_numeric_array",
                "columns": key["scope"]["covariates"],
                "requirement": "Computed by the frozen encoder; mismatched chain descriptors are rejected.",
            },
            "group": {
                "level": key["scope"]["grouping"]["level"],
                "requirement": "Retain source patents and antibody identity; no invented independence.",
            },
        }
    elif key["plugin_id"] == "generated_assay.v1":
        from agent.rsi.engine.generated_assay import INPUT_FIELDS

        fields = list(INPUT_FIELDS)
        contract = {
            "kind": "protein_substrate_assay",
            "representation": key["scope"]["representation"],
            "sequence": "Complete variable-length protein sequence; repeated proteins may have different measured substrates or conditions.",
            "values": {
                "type": "ordered_numeric_array",
                "columns": key["scope"]["covariates"],
                "requirement": "Use the frozen encoder and column units; identifiers are metadata, not features.",
            },
            "group": {
                "level": key["scope"]["grouping"]["level"],
                "requirement": "Preserve source protein and publication identities; do not invent independence.",
            },
        }
    elif contextual:
        fields = ["id", "protein", "group", "values"]
        contract = {
            "kind": "measured_context",
            "protein": key["scope"]["protein"],
            "values": {
                "type": "ordered_numeric_array",
                "columns": key["scope"]["covariates"],
                "requirement": "One finite value per column, in the listed order, units and inclusive bounds.",
            },
            "group": {
                "level": key["scope"]["grouping"]["level"],
                "requirement": "Use the recorded sample-group identity; do not invent a group to imply independence.",
            },
        }
    else:
        featured = key["plugin_id"] == "generated_features.v1"
        fields = ["id", "sequence", "features"] if featured else ["id", "sequence"]
        contract = {"kind": "sequence_with_features" if featured else "sequence"}
    return {
        **public_card(entry_id, key, verified["origin"], verified_now=True),
        "validation_metrics": verified["packet"]["assessment"]["metrics"],
        "input_fields": fields,
        "input_contract": {"schema": "evolved_model_inputs.v1", "target_required": False, **contract},
        "maximum_prediction_rows": 256,
        "evaluation_scope": "Frozen task validation; this does not establish performance on arbitrary proteins.",
    }


def predict_model(library, entry_id, inputs, scope, *, session_id=None, session_binding="tool_request"):
    from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed
    assert_tool_call_allowed('predict_with_evolved_model', {'entry_id':entry_id})
    _require(isinstance(inputs, list) and 1 <= len(inputs) <= 256, "Predict accepts 1 to 256 input rows")
    _require(isinstance(scope, dict), "Explicit model applicability scope required")
    _require(len(json.dumps(inputs, allow_nan=False).encode()) <= 1024 * 1024, "Input batch is too large")
    _require(all(isinstance(row, dict) for row in inputs), "Inputs must be records")
    if session_id is None:
        session_id = "agent_tool_request_" + uuid.uuid4().hex
        session_binding = "tool_request"
    _require(isinstance(session_id, str) and 0 < len(session_id) <= 256, "Invalid host session")
    result = library.predict(entry_id, inputs, scope, session_id=session_id)
    return {
        **result,
        "session_binding": session_binding,
        "scientific_evidence": "Ordinary Agent inference, not a new autonomous model or a new reuse experiment.",
    }


def dispatch(action, *, session_id=None, session_binding="tool_request", **arguments):
    """Shared LangChain/MCP boundary: reject online calls and keep internal paths private."""
    if action not in TOOL_NAMES:
        raise ValueError("Unknown evolved model tool")
    try:
        assert_tool_allowed(TOOL_NAMES[action])
    except RuntimeError as exc:
        return {"success": False, "error_code": "runtime_mode_disabled", "error": str(exc)}
    try:
        library = configured_library()
        if action == "list":
            result = list_models(library, **arguments)
        elif action == "describe":
            result = describe_model(library, **arguments)
        else:
            result = predict_model(library, **arguments, session_id=session_id, session_binding=session_binding)
        return {"success": True, **result}
    except Exception as exc:
        return {
            "success": False,
            "error_code": "model_unavailable_or_invalid_input",
            "error_type": type(exc).__name__,
            "error": "The model evidence or input could not be validated. Check the entry and its applicability scope.",
        }
