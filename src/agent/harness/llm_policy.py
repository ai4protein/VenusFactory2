"""Deployment-specific LLM publication gates, shared by APIs and workers."""
import hashlib
import json
from mcp_tools.tools.capability_policy import CapabilityStore


def route_key(model, base_url):
    identity = json.dumps([base_url.rstrip('/'), model], separators=(',', ':'))
    return 'llm:' + hashlib.sha256(identity.encode()).hexdigest()


def listed(model, base_url, default=True, *, rows=None):
    rows = CapabilityStore().rows() if rows is None else rows
    return bool(rows.get(route_key(model, base_url), {}).get('enabled', default))


def assert_listed(model, base_url, default=True):
    if not listed(model, base_url, default):
        raise ValueError('This LLM has been unpublished by the administrator')
