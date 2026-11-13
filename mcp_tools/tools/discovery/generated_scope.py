"""One paged registry view for public and owner-authorized Scientist capabilities."""

from __future__ import annotations

from agent.harness import capability_catalog as catalog
from agent.harness import research_workspace as runtime
from backend.web_v2.research_capability_access import private_libraries
from mcp_tools.models.catalog_snapshot import catalog_snapshot
from mcp_tools.tools import capability_policy
from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed, is_tool_allowed


class GeneratedScope:
    def __init__(self):
        self.private = {item.path: item for item in private_libraries()}
        self.sources = {runtime.LIBRARY: {"public_only": True}}
        self.sources.update({path: {"included": tuple(grant.deliveries)} for path, grant in self.private.items()})
        self.origins = {}

    def references(self, rank):
        """Rank IDs only; schemas, evidence and model weights stay unloaded."""
        # Snapshot this inventory operation only; the next request and the
        # invocation gate read current administrator settings again.
        settings = capability_policy.CapabilityStore().catalog_rows()
        result = []
        for library, options in self.sources.items():
            for identity, score in catalog.ranked_references(library, rank, **options):
                if identity in self.origins or not settings.get("model:capability:" + identity, {}).get("enabled", True):
                    continue
                self.origins[identity] = library
                result.append((identity, score))
        return result

    def cards(self, identities):
        # A session can be deleted or reassigned between ranking and display.
        current = {item.path: item for item in private_libraries()}
        result = {}
        for library, options in self.sources.items():
            selected = [identity for identity in identities if self.origins.get(identity) == library]
            if library != runtime.LIBRARY:
                if library not in current:
                    continue
                options = {"included": tuple(current[library].deliveries)}
            result.update(catalog.discovery_cards(library, selected, **options))
        return result

    def library_for(self, identity):
        assert_tool_call_allowed("run_generated_capability", {"capability_id": identity})
        if catalog.discovery_cards(runtime.LIBRARY, [identity], public_only=True):
            return runtime.LIBRARY
        for grant in private_libraries():
            library = grant.path
            if identity in grant.deliveries:
                grant.verify(identity)
                if catalog.discovery_cards(library, [identity], included=tuple(grant.deliveries)):
                    return library
        raise ValueError("Model unavailable. Search for an accessible model.")


@catalog_snapshot
def search(query="", limit=10, offset=0):
    if not isinstance(query, str) or len(query) > 200:
        raise ValueError("Use a search query of at most 200 characters")
    if type(limit) is not int or not 1 <= limit <= 50 or type(offset) is not int or offset < 0:
        raise ValueError("Use limit 1-50 and a nonnegative offset")
    # A search entry point can remain enabled while execution is switched off.
    # Match the progressive catalog without scanning any private/public registry.
    if not is_tool_allowed("run_generated_capability"):
        return {"total": 0, "items": [], "next_offset": None,
                "next_step": "Search for an enabled capability."}
    scope = GeneratedScope()
    if not scope.private:
        return runtime.search_capabilities(query, limit, library=runtime.LIBRARY, offset=offset)
    terms = query.casefold().split()
    references = scope.references(lambda name, text, kind: 1 if all(term in text for term in terms) else None)
    matches = sorted(identity for identity, rank in references if rank is not None)
    selected = matches[offset : offset + limit]
    cards = scope.cards(selected)
    return {
        "total": len(matches),
        "items": [cards[key] for key in selected if key in cards],
        "next_offset": offset + limit if offset + limit < len(matches) else None,
        "next_step": "Describe a selected ID for its complete scope, interface and review evidence before running it.",
    }


def describe(identity):
    return runtime.describe_capability(identity, library=GeneratedScope().library_for(identity))


async def run(identity, inputs):
    return await runtime.run_capability(identity, inputs, library=GeneratedScope().library_for(identity))
