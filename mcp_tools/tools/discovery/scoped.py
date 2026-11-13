"""Progressive model interfaces for both Scientist registry generations."""

from __future__ import annotations

import re
from copy import deepcopy

from mcp_tools.models.evolved.discovery import ScopedModelCatalog
from mcp_tools.tools import runtime_tool_policy

from .matching import query_groups, relevance


class ScopedScientistCatalog:
    category = "rsi"
    prefix = "scientist_capability:"
    required_tools = ("search_generated_capabilities", "describe_generated_capability", "run_generated_capability")

    def __init__(self, mapping, allowed):
        self.legacy = ScopedModelCatalog(mapping, allowed)
        self.tools = {tool.name: (name, tool) for name, tool in mapping.items() if tool.name in self.required_tools}
        self.allowed = allowed
        self.bindings = {}

    def generated_name(self, name):
        return isinstance(name, str) and re.fullmatch(self.prefix + r"[a-f0-9]{64}", name) is not None

    def owns(self, name):
        return self.generated_name(name) or self.legacy.owns(name)

    def generated_available(self):
        try:
            for name in self.required_tools:
                if name not in self.tools or not self.allowed(self.tools[name][1]):
                    return False
                # Resolve the current guard on every request; never retain a
                # temporary startup/test policy function in a cached catalog.
                runtime_tool_policy.assert_tool_allowed(name)
        except (ValueError, RuntimeError):
            return False
        return True

    def available(self, name=None):
        if name is None:
            return self.generated_available() or self.legacy.available()
        return self.generated_available() if self.generated_name(name) else self.legacy.available(name)

    def generated_page(self, query, limit, offset):
        if not self.generated_available():
            return None
        from . import generated_scope

        groups = query_groups(query)
        scope = generated_scope.GeneratedScope()

        def rank(name, text, kind):
            return relevance(groups, name, text + (" predict prediction" if kind == "model" else ""))

        matches = [(identity, score) for identity, score in scope.references(rank) if score is not None]
        matches.sort(key=lambda item: (-item[1], item[0]))
        selected = [identity for identity, _ in matches[offset : offset + limit]]
        cards = scope.cards(selected)
        return {
            "total": len(matches),
            "tools": [
                {
                    "name": self.prefix + identity,
                    "description": (cards[identity]["name"] + ". " + cards[identity]["description"])[:180],
                }
                for identity in selected
                if identity in cards
            ],
        }

    def search(self, query, limit, offset):
        page = self.generated_page(query, limit, offset)
        if not page or not page["total"]:
            return self.legacy.search(query, limit, offset)
        remaining = limit - min(limit, max(0, page["total"] - offset))
        older = self.legacy.search(query, max(1, remaining), max(0, offset - page["total"]))
        total = page["total"] + (older["total"] if older else 0)
        tools = page["tools"] + (older["tools"][:remaining] if older and remaining else [])
        return {
            "level": "capabilities",
            "category": self.category,
            "match_mode": "registered_models",
            "tools": tools,
            "total": total,
            "next_offset": offset + limit if offset + limit < total else None,
            "records_searched": True,
            "next_step": "Inspect one returned model with describe_tool(name), then call_tool(name, arguments).",
        }

    def describe(self, name):
        if not self.generated_name(name):
            return self.legacy.describe(name)
        if not self.generated_available():
            raise ValueError("Model unavailable. Search for an enabled model.")
        from . import generated_scope

        identity = name[len(self.prefix) :]
        card = generated_scope.describe(identity)
        self.bindings[name] = deepcopy(card["input_schema"])
        return {
            "name": name,
            "description": card["description"],
            "call_via": "call_tool",
            "inputSchema": {
                "type": "object",
                "properties": {"inputs": card["input_schema"]},
                "required": ["inputs"],
                "additionalProperties": False,
            },
            "output_schema": card["output_schema"],
            **{
                key: card[key]
                for key in ("scope", "kind", "version", "author", "author_email", "visibility", "review_status")
                if key in card
            },
            "next_step": "Call this same name with inputs that satisfy the described contract and scientific scope.",
        }

    def resolve(self, name, arguments):
        if not self.generated_name(name):
            return self.legacy.resolve(name, arguments)
        if not self.generated_available() or name not in self.bindings:
            raise ValueError("Inspect an enabled model before calling it.")
        if not isinstance(arguments, dict) or set(arguments) != {"inputs"}:
            raise ValueError("Provide inputs only; model identity and ownership are assigned by the host.")
        from jsonschema.exceptions import ValidationError

        from agent.harness.capability_review import schema_validator

        from . import generated_scope

        try:
            schema_validator(self.bindings[name]).validate(arguments["inputs"])
        except ValidationError:
            raise ValueError(
                "Inputs do not match the selected model's contract. Check describe_tool for required fields and types."
            ) from None
        identity = name[len(self.prefix) :]
        generated_scope.GeneratedScope().library_for(identity)
        key, tool = self.tools["run_generated_capability"]
        return key, tool, {"capability_id": identity, "inputs": arguments["inputs"]}
