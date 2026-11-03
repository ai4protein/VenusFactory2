"""Scientific review never grants public registration; only an operator can publish."""
import json
import pytest
from agent.harness import research_workspace as rw
from agent.harness.curated_publication import publish_candidate
from agent.harness.capability_review import review_proposal
from test_capability_admission import candidate, spec, reviewer


@pytest.mark.asyncio
async def test_private_review_operator_publication_email_and_integrity(tmp_path,monkeypatch):
    monkeypatch.setenv("VENUS_RUNTIME_ROOT",str(tmp_path/"runtime"))
    from mcp_tools.tools import runtime_tool_policy as policy
    monkeypatch.setattr(policy,"is_online_mode",lambda:False)
    shared=tmp_path/"shared";monkeypatch.setattr(rw,"LIBRARY",shared)
    service=await candidate(tmp_path,spec());bundle=service.proposals[0]
    review=await review_proposal(bundle,reviewer=reviewer,control=tmp_path/"review",runner=rw.run_python)
    assert review["status"] == "approved"
    private=rw.publish_proposals([bundle],"goal",tmp_path/"any_private_directory")[0]
    assert private["visibility"] == "private" and "author_email" not in private
    assert rw.capability_cards(shared) == []
    with pytest.raises(ValueError,match="operator"):
        rw.publish_proposals([bundle],"goal",shared)
    with pytest.raises(ValueError,match="email"):
        publish_candidate(bundle,"goal",library=shared,author_email="invalid",curator="fixture")
    monkeypatch.setattr(policy,"is_online_mode",lambda:True)
    with pytest.raises(ValueError,match="operator"):
        publish_candidate(bundle,"goal",library=shared,author_email="author@example.org",curator="fixture")
    monkeypatch.setattr(policy,"is_online_mode",lambda:False)
    public=publish_candidate(bundle,"goal",library=shared,author_email="author@example.org",curator="fixture")
    assert public["visibility"] == "public"
    assert rw.capability_cards(shared)[0]["author_email"] == "author@example.org"
    assert rw.describe_capability(public["id"],shared)["author_email"] == "author@example.org"
    # A mere successful run or legacy 'public' flag never admits an old entry.
    with rw.safe_file(shared/public["id"],"publication.json").open("w") as stream:
        json.dump({},stream)
    with pytest.raises(ValueError,match="Publication"):
        rw.admitted_manifest(public["id"],shared)


def test_publication_tools_are_not_available_in_either_mode(tmp_path,monkeypatch):
    monkeypatch.setenv("VENUS_RUNTIME_ROOT",str(tmp_path))
    from mcp_tools.tools import runtime_tool_policy as policy
    for online in (True,False):
        monkeypatch.setattr(policy,"is_online_mode",lambda:online)
        assert not policy.is_tool_allowed("register_trained_model")


def test_browser_model_visibility_cannot_publish(tmp_path,monkeypatch):
    from backend.web_v2.model_platform import Platform
    monkeypatch.setattr(Platform,"owned",lambda *args: {})
    with pytest.raises(ValueError,match="operator"):
        Platform(tmp_path).visibility("user","irrelevant","public")
