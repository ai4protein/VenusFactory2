"""Expose individual registered models without a second inference backend.

A selected interface binds an immutable model and its described scope for this
discovery session. Execution still resolves to the ordinary queued prediction tool.
"""
from __future__ import annotations

from copy import deepcopy
import re

from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed


class ScopedModelCatalog:
    category = "rsi"
    prefix = "scientist_model:"
    required_tools = ("list_evolved_models", "describe_evolved_model", "predict_with_evolved_model")

    def __init__(self, mapping, allowed):
        self.tools = {tool.name: (name, tool) for name, tool in mapping.items()
                      if tool.name in self.required_tools}
        self.allowed = allowed
        self.bindings = {}

    def owns(self, name):
        return isinstance(name, str) and re.fullmatch(self.prefix + r"[0-9a-f]{64}", name) is not None

    def available(self, name=None):
        if any(key not in self.tools or not self.allowed(self.tools[key][1]) for key in self.required_tools):
            return False
        try:
            assert_tool_call_allowed("predict_with_evolved_model",
                                     {"entry_id": name[len(self.prefix):]} if name else {})
        except (ValueError, RuntimeError):
            return False
        return True

    @staticmethod
    def _read(action, **arguments):
        from .service import dispatch
        result = dispatch(action, **arguments)
        if not result.get("success"):
            raise ValueError("Registered model unavailable. Search again or select another enabled capability.")
        return result

    def search(self, query, limit, offset):
        if not self.available():
            return None
        page = self._read("list", query=query, limit=limit, offset=offset)
        return {
            "level": "capabilities", "category": self.category, "match_mode": "registered_models",
            "tools": [{"name": self.prefix + row["entry_id"],
                       "description": (row["name"] + ". " + row["description"])[:180]}
                      for row in page["models"] if self.available(self.prefix + row["entry_id"])],
            "total": page["matching_entries"], "next_offset": page["next_offset"],
            "records_searched": True,
            "next_step": "Inspect one returned model with describe_tool(name), then call_tool(name, arguments).",
        }

    def describe(self, name):
        if not self.owns(name) or not self.available(name):
            raise ValueError("Model unavailable. Search for an enabled model.")
        entry = name[len(self.prefix):]
        card = self._read("describe", entry_id=entry)
        if not self.available(name):
            raise ValueError("Model unavailable. Search for an enabled model.")
        # Snapshot only metadata that describe has actually verified. Neither
        # search nor a guessed identifier creates an executable binding.
        self.bindings[name] = {"entry_id": entry, "scope_ref": card["scope_ref"]}
        from .contract import PredictionArguments
        field = deepcopy(PredictionArguments.model_json_schema()["properties"]["inputs"])
        return {
            "name": name, "description": "Predict with the selected model using its input contract and scope below.",
            "call_via": "call_tool",
            "inputSchema": {"type": "object", "properties": {"inputs": field},
                            "required": ["inputs"], "additionalProperties": False},
            "scope": card["scope"], "input_fields": card["input_fields"],
            "input_contract": card["input_contract"],
            "validation_metrics": card["validation_metrics"],
            "validation_metrics_status": card["validation_metrics_status"],
            "verified_now": card["verified_now"], "evaluation_scope": card["evaluation_scope"],
            "next_step": "Call this same name with inputs. Its model and scope are bound to the inspected interface.",
        }

    def resolve(self, name, arguments):
        if not self.available(name):
            raise ValueError("Model unavailable. Search for an enabled model.")
        if name not in self.bindings:
            raise ValueError("Inspect the selected model before calling it.")
        if not isinstance(arguments, dict) or set(arguments) != {"inputs"}:
            raise ValueError("This model interface accepts inputs only; its model, scope and session are assigned by the host.")
        key, tool = self.tools["predict_with_evolved_model"]
        # No scientific input is invented or normalized. The downstream contract,
        # queue and numerical replay recheck the exact binding and supplied rows.
        return key, tool, {**self.bindings[name], "inputs": arguments["inputs"]}
