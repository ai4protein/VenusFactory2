from research_reviewer_fixture import reviewer_catalog as _reviewer_catalog  # noqa: F401
import pytest
from backend.web_v2 import research_longform as lf


@pytest.mark.parametrize("value,expected", [(None,48),("32",32),("0",1),("999",96),("invalid",48)])
def test_new_report_budget_is_configurable_and_bounded(monkeypatch,value,expected):
    monkeypatch.delenv("VENUS_REPORT_MODEL_CALL_LIMIT",raising=False)
    if value is not None:
        monkeypatch.setenv("VENUS_REPORT_MODEL_CALL_LIMIT",value)
    assert lf.report_model_call_limit()==expected


@pytest.mark.parametrize("saved,configured,expected", [(18,"48",18),(48,"24",24),(7,"96",7)])
def test_resume_never_increases_a_frozen_budget(monkeypatch,saved,configured,expected):
    monkeypatch.setenv("VENUS_REPORT_MODEL_CALL_LIMIT",configured)
    assert lf.report_model_call_limit({"model_call_limit":saved})==expected


@pytest.mark.parametrize("saved", [None,True,0,"96"])
def test_legacy_or_invalid_checkpoint_uses_legacy_ceiling(monkeypatch,saved):
    monkeypatch.setenv("VENUS_REPORT_MODEL_CALL_LIMIT","48")
    checkpoint={} if saved is None else {"model_call_limit":saved}
    assert lf.report_model_call_limit(checkpoint)==18


@pytest.mark.asyncio
async def test_exhausted_saved_budget_cannot_resume_under_a_higher_deployment_limit(monkeypatch):
    import copy
    from unittest.mock import patch
    checkpoint={"workflow":"specialist_research_v2","phase":"incomplete","collection_complete":True,
        "model_calls":3,"model_call_limit":3,"sources":[],"steps":[],"chapters":[],
        "report":"","partial":True,"collection_partial":True}
    checkpoint["chapters"]=[{"id":identity,"title":title,"status":"pending","draft":"",
        "source_ids":[],"attempts":0} for identity,title,*_ in lf.CHAPTERS]
    async def forbidden(*args,**kwargs):
        raise AssertionError("An exhausted report must not issue a provider request")
    monkeypatch.setenv("VENUS_REPORT_MODEL_CALL_LIMIT","48")
    with patch.object(lf.research,"complete",side_effect=forbidden):
        result=await lf.run_long_report({"model_id":"fixture","analyses":["properties"],"lang":"en"},lambda x:None,checkpoint=copy.deepcopy(checkpoint))
    assert result["model_calls"]==3 and result["model_call_limit"]==3
    assert result["phase"]=="incomplete"
