"""RSI cases never re-enter the executable Model Plaza, including old IDs."""
import pytest
from backend.web_v2.model_platform import Platform
from mcp_tools.models.evolved import service


@pytest.mark.parametrize("identity", ["scientist:stale", "capability:reviewed"])
def test_old_rsi_model_details_do_not_read_or_replay_evidence(tmp_path, monkeypatch, identity):
    p = Platform(tmp_path)
    monkeypatch.setattr(p, "shared", lambda: pytest.fail("Old RSI ID read executable catalog"))
    monkeypatch.setattr(service, "describe_model", lambda *a: pytest.fail("Replayed retired model"))
    with pytest.raises(FileNotFoundError): p.model("owner", identity, detail=True)
    with pytest.raises(FileNotFoundError): p.predict("owner", identity, {"kind":"table","rows":[]})


@pytest.mark.parametrize("local", [True, False])
def test_shared_catalog_never_loads_rsi_but_keeps_factory_models(tmp_path, monkeypatch, local):
    from agent.harness import research_workspace
    from mcp_tools.tools import runtime_tool_policy
    from mcp_tools.models.factory.training import model_registry
    p = Platform(tmp_path)
    monkeypatch.setattr(runtime_tool_policy, "is_local_mode", lambda: local)
    monkeypatch.setattr(p, "official", lambda: [{"id":"builtin:example"}])
    monkeypatch.setattr(model_registry, "list_trained_models", lambda: {"models":[]})
    monkeypatch.setattr(service, "configured_library", lambda: pytest.fail("Model Plaza scanned RSI library"))
    monkeypatch.setattr(research_workspace, "capability_cards", lambda: pytest.fail("Model Plaza scanned RSI programs"))
    assert p.shared() == ([{"id":"builtin:example"}], [])
