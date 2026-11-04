"""Public metadata only; credentials and upstream requests stay off this API."""
from agent.harness.catalog import resolve_credential
from agent.harness.free_catalog import free_routes
from agent.harness.llm_policy import listed, route_key
from agent.harness.provider_config import online
from mcp_tools.tools.capability_policy import CapabilityStore


def inventory():
    from backend.web_v2.agent_execution import catalog
    cat = catalog(for_admin=True)
    saved = CapabilityStore().rows()
    free = {(r['model'], r['base_url'], r['protocol']) for r in free_routes(include_unlisted=True)[0]}
    items = []
    for option in cat.models.values():
        binding = option.binding
        key = route_key(binding.model, binding.base_url)
        is_free = (binding.model, binding.base_url.rstrip('/'), binding.protocol) in free
        try:
            resolve_credential(binding, option.credential_toml)
            configured = True
        except (OSError, ValueError, KeyError):
            configured = False
        allowed = not online() or is_free
        enabled = listed(binding.model, binding.base_url, option.listed_by_default, rows=saved)
        items.append(dict(id=option.id, key=key, model=binding.model, provider=binding.provider,
                          free=is_free, configured=configured, enabled=enabled,
                          available=enabled and configured and allowed,
                          can_publish=configured and allowed,
                          reason='free_model_required' if not allowed else 'credential_unavailable' if not configured else None,
                          revision=saved.get(key, {}).get('revision', 0)))
    return dict(mode='online' if online() else 'local', items=items)


def publish(key, enabled, revision, actor):
    row = next((r for r in inventory()['items'] if r['key'] == key), None)
    if row is None:
        raise LookupError('LLM not found')
    if enabled and not row['can_publish']:
        raise PermissionError('Configure an API key and an allowed model before publishing')
    return CapabilityStore().set(key, enabled, revision, {}, actor)
