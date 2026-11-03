"""Search platform models without loading model weights or input schemas into prompts."""
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
