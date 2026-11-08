import pytest
from backend.web_v2.model_platform import Platform
from mcp_tools.models.evolved import service

def test_stale_model_detail_remains_readable_but_cannot_run(tmp_path,monkeypatch):
    p=Platform(tmp_path)
    card={"id":"scientist:stale","name":"Published model","adapter":"scientist",
          "source":"scientist","category":"function","input_kind":"table","workflow":True}
    monkeypatch.setattr(p,"shared",lambda:([card],[]))
    monkeypatch.setattr(service,"configured_library",lambda:object())
    def changed(*args):
        raise ValueError("plugin implementation changed after policy freeze")
    monkeypatch.setattr(service,"describe_model",changed)
    result=p.model("owner","scientist:stale",detail=True)
    assert result["available"] is False
    assert result["disabled_reason"]=="needs_revalidation"
    assert result["contract"]["verified_now"] is False
    with pytest.raises(ValueError,match="revalidation"):p.require_available(result)

def test_evolved_search_excludes_incompatible_publications(monkeypatch):
    from mcp_tools.tools.capability_policy import CapabilityStore
    monkeypatch.setattr(CapabilityStore,"enabled",lambda *args:True)
    cards=[{"entry_id":"old","stage":"test","runtime_compatible":False},
           {"entry_id":"current","stage":"test","runtime_compatible":True}]
    for card in cards:
        card.update(scope={"protein":"fixture","target":"activity","assay":"measurement"},
                    interface="generated_python.v1", verified_now=False)
    monkeypatch.setattr(service,"catalog_cards",lambda _: (cards,{"indexed_entries":2,"counting_rule":"fixture"}))
    result=service.list_models(object())
    assert [x["entry_id"] for x in result["models"]]==["current"]


def test_known_stale_card_does_not_replay_full_evidence_on_detail(tmp_path, monkeypatch):
    p = Platform(tmp_path)
    card = {"id": "scientist:stale", "name": "Archived model", "adapter": "scientist",
            "source": "scientist", "category": "function", "input_kind": "table",
            "workflow": True, "validation_status": "needs_revalidation"}
    monkeypatch.setattr(p, "shared", lambda: ([card], []))
    def unexpected(*args):
        raise AssertionError("Known incompatible evidence must not be replayed on a detail request")
    monkeypatch.setattr(service, "describe_model", unexpected)
    detail = p.model("owner", card["id"], detail=True)
    assert not detail["available"]
    assert detail["contract"]["verified_now"] is False
