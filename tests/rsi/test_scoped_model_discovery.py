"""Persisted conversational RSI selectors cannot bypass case-only publication."""
import pytest
from agent.harness.session_boundary import mcp_input_schema
from agent.harness.tool_disclosure import ToolDisclosure, DisclosureError
from mcp_tools.models.evolved import discovery, service
from mcp_tools.models.evolved.tools_agent import EVOLVED_MODEL_TOOLS


def index(allowed=lambda tool: True):
    mapping = {tool.name: tool for tool in EVOLVED_MODEL_TOOLS}
    models = discovery.ScopedModelCatalog(mapping, allowed)
    return ToolDisclosure(mapping, schema=mcp_input_schema, allowed=allowed, model_catalog=models)


@pytest.mark.parametrize("online", [True, False])
def test_old_model_binding_is_unavailable_even_if_adapter_callback_allows_it(monkeypatch, online):
    from mcp_tools.tools import runtime_tool_policy
    monkeypatch.setattr(runtime_tool_policy, "is_online_mode", lambda: online)
    monkeypatch.setattr(service, "dispatch", lambda *a, **k: pytest.fail("RSI registry accessed"))
    catalog = index()
    name = discovery.ScopedModelCatalog.prefix + "a"*64
    catalog.model_catalog.bindings[name] = {"entry_id":"a"*64, "scope_ref":"old"}
    assert not catalog.model_catalog.available(name)
    assert catalog.model_catalog.search("protein", 10, 0) is None
    with pytest.raises(ValueError, match="unavailable"): catalog.model_catalog.describe(name)
    with pytest.raises(ValueError, match="unavailable"): catalog.model_catalog.resolve(name, {"inputs": []})


def test_navigation_adapter_import_does_not_load_scientific_runtime():
    import subprocess, sys
    script = "import sys; import mcp_tools.models.evolved.discovery; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, '-c', script], check=True)
