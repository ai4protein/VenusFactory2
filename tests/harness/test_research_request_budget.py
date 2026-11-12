"""SC corrections may use reserved requests without increasing the episode cap."""
import json
from types import SimpleNamespace

import pytest

from agent.harness.autonomous_research import AutonomousResearch
from agent.harness.research_request_budget import request_ceiling


@pytest.mark.parametrize("role,initial,attempts", [
    ("SC", 13, 2), ("REVIEW", 13, 2), ("CB", 12, 1), ("PI", 12, 1),
    ("SC", 15, 1), ("REVIEW", 15, 1),
])
@pytest.mark.asyncio
async def test_role_repairs_use_only_their_remaining_real_requests(tmp_path, role, initial, attempts):
    relay = SimpleNamespace(calls=initial)
    observed = []
    original = {"objective": "Evaluate the supplied protein measurements"}

    async def consult(actual_role, context):
        assert actual_role == role
        assert context["objective"] == original["objective"]
        relay.calls += 1
        observed.append(context)
        if len(observed) == 1:
            return "malformed response"
        assert context["format_repair"]["previous_response"] == "malformed response"
        return '{"decision":"needs_revision"}'

    runner = AutonomousResearch(None, "unused", {}, consult=consult)
    pending = runner.structured(role, original, SimpleNamespace(budget=SimpleNamespace(model_calls=16)),
                                relay, tmp_path, json.loads)
    if attempts == 2:
        assert await pending == {"decision": "needs_revision"}
    else:
        with pytest.raises(ValueError):
            await pending
    assert len(observed) == attempts
    assert relay.calls == initial + attempts <= 16
    assert original == {"objective": "Evaluate the supplied protein measurements"}


@pytest.mark.asyncio
async def test_review_cannot_loop_on_malformed_responses_even_with_extra_budget(tmp_path):
    relay = SimpleNamespace(calls=0)

    async def consult(role, context):
        relay.calls += 1
        return "still invalid"

    runner = AutonomousResearch(None, "unused", {}, consult=consult)
    with pytest.raises(ValueError):
        await runner.structured("SC", {"objective": "Review evidence"},
                                SimpleNamespace(budget=SimpleNamespace(model_calls=64)),
                                relay, tmp_path, json.loads)
    assert relay.calls == 2
    assert request_ceiling("SC", 64) == request_ceiling("REVIEW", 64) == 64
    assert request_ceiling("PI", 64) == request_ceiling("CB", 64) == 61
