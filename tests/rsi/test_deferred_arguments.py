from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from agent.graph.execution.prepare import materialize_deferred_input
from agent.graph.planning.plan import _filter_unparameterized_steps


class Args(BaseModel):
    source_code: str
    rationale: str


def ctx(materializer=None, provided=None, enabled=True):
    return SimpleNamespace(
        tool=SimpleNamespace(name="code", args_schema=Args, metadata={"deferred_arguments": enabled}),
        merged_tool_input=provided or {},
        chains={"materialize_tool_arguments": materializer},
    )


def test_planner_defers_only_with_explicit_plugin_and_materializer():
    step = {"step": 1, "tool_name": "code", "tool_input": {}}
    tool = ctx().tool
    assert _filter_unparameterized_steps([step], [tool], "en")[0] == []
    assert _filter_unparameterized_steps([step], [tool], "en", allow_deferred_arguments=True)[0] == [step]
    tool.metadata = {}
    assert _filter_unparameterized_steps([step], [tool], "en", allow_deferred_arguments=True)[0] == []


@pytest.mark.asyncio
async def test_materialized_values_are_typed_and_empty_values_rejected():
    async def fill(context):
        return {"source_code": "full source", "rationale": "reason"}

    assert await materialize_deferred_input(ctx(fill)) == {"source_code": "full source", "rationale": "reason"}

    async def empty(context):
        return {"source_code": "", "rationale": "reason"}

    with pytest.raises(ValueError, match="empty"):
        await materialize_deferred_input(ctx(empty))

    async def extra(context):
        return {"source_code": "source", "rationale": "reason", "shell": "arbitrary"}

    with pytest.raises(ValueError, match="undeclared"):
        await materialize_deferred_input(ctx(extra))


@pytest.mark.asyncio
async def test_complete_or_ordinary_arguments_do_not_trigger_generation():
    async def forbidden(context):
        raise AssertionError("unneeded model call")

    args = {"source_code": "source", "rationale": "reason"}
    assert await materialize_deferred_input(ctx(forbidden, args)) == args
    assert await materialize_deferred_input(ctx(forbidden, {"legacy": 1}, False)) == {"legacy": 1}
    with pytest.raises(ValueError, match="unavailable"):
        await materialize_deferred_input(ctx())


@pytest.mark.asyncio
async def test_typed_dependency_objects_require_materialization():
    async def select(context):
        return {"source_code": "complete implementation", "rationale": "actual dependency inspected"}

    got = await materialize_deferred_input(
        ctx(select, {"source_code": {"upstream": "object"}, "rationale": "existing"})
    )
    assert got["source_code"] == "complete implementation"
