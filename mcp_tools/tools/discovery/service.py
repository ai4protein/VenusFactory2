"""Bounded, current capability discovery; inference stays with its registered tool."""

from __future__ import annotations

import json
import logging
from collections import Counter
from itertools import zip_longest

from mcp_tools.tools.runtime_tool_policy import is_local_mode
from .matching import query_groups, relevance
from .platform_models import platform_models
from mcp_tools.models.catalog_snapshot import catalog_snapshot

logger = logging.getLogger(__name__)


def available_tools():
    # Import after the hub has registered this tool to avoid a circular import.
    from mcp_tools.tools.tools_agent_hub import get_tools
    from mcp_tools.tools.runtime_tool_policy import filter_tools

    return filter_tools(get_tools())[0]


def model_cards():
    from mcp_tools.models.evolved import service as evolved

    cards, _ = evolved.catalog_cards(evolved.configured_library())
    return cards


@catalog_snapshot
def search(query="", kind="all", stage="", limit=10, offset=0, *, tool_catalog=None):
    if not isinstance(query, str) or len(query) > 200:
        return {"success": False, "error": "Use a search query of at most 200 characters."}
    if not isinstance(kind, str) or kind not in {"all", "tools", "models"}:
        return {"success": False, "error": "Choose kind: all, tools or models."}
    if not isinstance(stage, str) or len(stage) > 80:
        return {"success": False, "error": "Invalid model stage."}
    if type(limit) is not int or not 1 <= limit <= 30 or type(offset) is not int or offset < 0:
        return {"success": False, "error": "Use limit 1–30 and a nonnegative offset."}

    groups = query_groups(query)
    tool_results, model_results = [], []
    from mcp_tools.tools.runtime_tool_policy import is_tool_allowed

    from mcp_tools.scheduler.catalog import tool_family
    tools = []
    catalog = tool_catalog if tool_catalog is not None else [
        {"name": t.name, "description": t.description, "parameters": t.tool_call_schema.model_json_schema(), "metadata": t.metadata, "model_family": tool_family(t.name, t)}
        for t in available_tools()
    ]
    tools = [t for t in catalog if t["name"] != "search_research_tools" and is_tool_allowed(t["name"])]
    for tool in tools if kind != "models" else []:
        name = tool["name"]
        description = tool.get("description") or ""
        score = relevance(groups, name.replace("_", " "), name + " " + description)
        if score is None:
            continue
        schema = tool["parameters"]
        from mcp_tools.scheduler.catalog import classify
        tool_results.append((score, name, {
            "kind": "tool",
            "model_family": tool.get("model_family") or tool_family(name, None),
            **classify(name, tool.get("metadata")),
            "name": name,
            "description": description[:600],
            "required_arguments": schema.get("required", []),
            "arguments": list(schema.get("properties", {})),
        }))

    platform = platform_models(query, [t["name"] for t in tools]) if kind != "tools" and not stage else []
    model_results.extend(platform)
    cards = []
    model_status = "not_requested"
    if kind != "tools":
        model_status = "available"
        if is_local_mode():
            try:
                cards = model_cards()
                model_status = "available"
            except Exception:
                logger.exception("Scientist model catalog could not be loaded")
                model_status = "unavailable"
    from mcp_tools.tools.capability_policy import CapabilityStore
    settings = CapabilityStore()
    cards = [
        card for card in cards
        if card.get("runtime_compatible", True)
        and settings.enabled("model:scientist:" + card["entry_id"])
        and is_tool_allowed("predict_with_evolved_model")
    ]
    if cards:
        from mcp_tools.models.evolved.service import discovery_card
    for card in cards:
        if stage and card["stage"] != stage:
            continue
        scope = card["scope"]
        title = " · ".join(str(scope.get(k, "")) for k in ("protein", "target"))
        score = relevance(groups, title, json.dumps(card, ensure_ascii=False))
        if score is None:
            continue
        model_results.append((score, card["entry_id"], {
            "kind": "scientist_model",
            "model_family": "rsi",
            **discovery_card(card),
        }))

    if kind != "tools" and not stage:
        from agent.harness.research_workspace import capability_cards
        for item in capability_cards():
            if not settings.enabled('model:capability:' + item['id']) or not is_tool_allowed('run_generated_capability'): continue
            score = relevance(groups, item["name"], item["name"] + " " + item["description"])
            if score is not None:
                model_results.append((score, item["id"], {"kind": "research_capability", "model_family": "rsi", "name": item["name"], "entry_id": item["id"], "description": item["description"], "validation": item["validation"], **{k:item[k] for k in ("author","author_email","version","scope","review_status") if k in item}, "describe": {"tool": "describe_generated_capability", "arguments": {"capability_id": item["id"]}}, "predict_tool": "run_generated_capability"}))

    order = lambda item: (-item[0], item[1])
    tool_results.sort(key=order)
    model_results.sort(key=order)
    if groups:
        combined = sorted(tool_results + model_results, key=order)
    else:
        combined = [item for pair in zip_longest(tool_results, model_results) for item in pair if item is not None]
    total = len(combined)
    return {
        "success": True,
        "query": query,
        "kind": kind,
        "results": [item[2] for item in combined[offset:offset + limit]],
        "matching_results": total,
        "matching_tools": len(tool_results),
        "matching_models": len(model_results),
        "available_tools": len(tools),
        "indexed_models": len(cards) + len(platform_models("", [t["name"] for t in tools])) if kind != "tools" else 0,
        "model_stages": dict(sorted(Counter(c["stage"] for c in cards).items())),
        "model_catalog_status": model_status,
        "next_offset": offset + limit if offset + limit < total else None,
        "model_usage": "Describe the selected entry, then predict with its exact scope and required inputs. Ask for missing inputs. Published protein models are callable scientific tools, separate from the chat language-model selector.",
    }
