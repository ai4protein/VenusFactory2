import pytest

from agent.harness.research_workspace import ResearchWorkspace
from agent.harness.session_boundary import mcp_input_schema


def service(tmp_path, steps=("inspect", "fit")):
    return ResearchWorkspace(tmp_path/"work", tmp_path/"control",
                             {"steps":[{"id":step} for step in steps]})


def test_checkpoint_schema_discloses_the_current_step_ids(tmp_path):
    work = service(tmp_path)
    tool = next(t for t in work.tools() if t.name == "submit_research_checkpoint")
    field = mcp_input_schema(tool)["properties"]["completed_steps"]
    assert field["items"]["enum"] == ["inspect","fit"]
    assert "never include titles or descriptions" in field["description"]


@pytest.mark.asyncio
async def test_descriptions_are_not_silently_accepted_as_completed_ids(tmp_path):
    work = service(tmp_path)
    await work.write_research_file("observed.json", '{"count":2}')
    with pytest.raises(ValueError) as error:
        await work.submit_research_checkpoint("Inspected input", ["inspect: Inspected all data"],
                                              ["observed.json"], [])
    assert '"inspect"' in str(error.value) and '"fit"' in str(error.value)
    assert "exact step IDs only" in str(error.value)
    assert not (tmp_path/"control/checkpoint.json").exists()
    result = await work.submit_research_checkpoint("Inspected input", ["inspect"], ["observed.json"], [])
    assert result["checkpointed"] and work.checkpoint["completed_steps"] == ["inspect"]
    assert result["artifacts"][0]["sha256"]


@pytest.mark.asyncio
async def test_partial_checkpoint_is_distinct_from_completed_work(tmp_path):
    work = service(tmp_path, ())
    tool = next(t for t in work.tools() if t.name == "submit_research_checkpoint")
    assert mcp_input_schema(tool)["properties"]["completed_steps"]["maxItems"] == 0
    result = await work.submit_research_checkpoint("Still inspecting supplied data", [], [], [])
    assert result["checkpointed"] and not work.checkpoint["completed_steps"]
    with pytest.raises(ValueError, match="brief summary"):
        await work.submit_research_checkpoint("", [], [], [])
