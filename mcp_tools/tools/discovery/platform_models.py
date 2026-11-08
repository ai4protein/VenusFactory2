"""Search platform models without loading model weights or input schemas into prompts."""
from copy import deepcopy

from mcp_tools.models.platform_catalog import official_cards
from mcp_tools.tools.capability_policy import canonical, model_tools
from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
from .matching import query_groups, relevance
from mcp_tools.models.catalog_snapshot import catalog_snapshot


@catalog_snapshot
def platform_models(query="", names=None):
    """Return bounded-search metadata; callers paginate. Availability is checked live."""
    mounted = {canonical(n) for n in names} if names is not None else None
    groups = query_groups(query)
    results = []
    for card in official_cards():
        if not card.get("available"):
            continue
        tools = sorted(n for n in model_tools(card)
                       if is_tool_allowed(n) and (mounted is None or n in mounted))
        if not tools:
            continue
        title = card["name"]
        text = " ".join(str(card.get(k, "")) for k in
                        ("name", "description", "model", "task", "dataset", "category", "family"))
        score = relevance(groups, title, text)
        if score is None:
            continue
        # Selector hints are not an execution contract. Read the actual tool schema next.
        selection = {k: card[k] for k in ("model", "task", "dataset", "repository") if card.get(k)}
        results.append((score, card["id"], {
            "kind": "platform_model", "entry_id": card["id"], "name": title,
            "description": str(card.get("description", ""))[:180],
            "category": card["category"], "model_family": card.get("family", card.get("model_type", "platform")),
            "tools": tools, "selection": selection,
        }))
    return results


def describe_model_choices(tool_name, input_schema, requested=()):
    """Bind current catalog choices to one selected tool's declared interface.

    Only description calls use this function. Search does not load schemas, and
    these selector hints never replace invocation-time permission checks.
    """
    result = {"inputSchema": deepcopy(input_schema)}
    live_models = {
        card["entry_id"]: card for _, _, card in platform_models("", [tool_name])
    }
    wanted = requested or list(live_models.values())
    properties = result["inputSchema"].get("properties", {})
    options = []
    selector_fields = {
        "model": ("model_name", "model"),
        "dataset": ("dataset_name", "dataset"),
        "task": ("task",),
    }
    for candidate in wanted[:20]:
        live = live_models.get(candidate["entry_id"])
        if not live:
            continue
        selection = live["selection"]
        preset = {}
        for field, keys in selector_fields.items():
            if field not in selection:
                continue
            for key in keys:
                if key in properties and (
                    not properties[key].get("enum")
                    or selection[field] in properties[key]["enum"]
                ):
                    preset[key] = selection[field]
                    break
        options.append({"name": candidate["name"], "arguments": preset})
    if not options:
        return result, set()

    model_keys = {
        key for option in options for key in option["arguments"]
        if key in {"model_name", "model"}
    }
    for key in model_keys:
        properties[key].pop("default", None)
        properties[key]["description"] = (
            "Explicit model selector. Use the requested model from matching_models; "
            "there is no implicit model choice."
        )
        required = result["inputSchema"].setdefault("required", [])
        if key not in required:
            required.append(key)
    result.update(
        scientific_inputs=[
            key for key in result["inputSchema"].get("required", []) if key not in model_keys
        ],
        model_count=len(wanted),
        matching_models=options,
        selection_rule=(
            "For the requested model, use its explicit arguments instead of generic defaults; "
            "ask for required scientific inputs only."
        ),
    )
    if len(wanted) > len(options):
        result["more_models"] = "Search by the requested model name to narrow the catalog."
    return result, model_keys
