from research_reviewer_fixture import reviewer_catalog as _reviewer_catalog  # noqa: F401
from research_source_review_fixture import with_source_review
import copy
import json
import pytest
from backend.web_v2 import research_longform as lf


@pytest.mark.asyncio
@pytest.mark.parametrize("collection_partial", [False, True, "legacy"])
async def test_retry_writes_only_unfinished_chapters_and_preserves_call_accounting(monkeypatch, collection_partial):
    checkpoint = {
        "workflow": "specialist_research_v2", "phase": "completed", "model_calls": 12,
        "collection_complete": True, "partial": True, "minimum_met": True, "chapters": [],
        "sources": [{"id": "S1", "kind": "literature", "title": "Evidence", "url": "https://example.org",
                     "text": "A public measurement", "evidence_level": "abstract"}],
        "protein": {"name": "Protein"}, "steps": [], "sections": [], "report": "Previous report",
    }
    if collection_partial != "legacy":
        checkpoint["collection_partial"] = collection_partial
    for i, (identity, title, *_) in enumerate(lf.CHAPTERS):
        checkpoint["chapters"].append({
            "id": identity, "title": title, "status": "partial" if i == 6 else "completed",
            "draft": f"Retained draft {i}. " + "Measured evidence " * 250 + "[S1]", "source_ids": ["S1"], "attempts": 2 if i == 6 else 1,
        })
    saved = copy.deepcopy(checkpoint)
    calls = []
    async def forbidden_collection(*args, **kwargs):
        raise AssertionError("Completed collection repeated")
    @with_source_review
    async def complete(model, messages, tokens, **_review_metadata):
        calls.append(messages)
        return "Measured evidence " * 260 + "[S1]", False
    monkeypatch.setattr(lf.research, "run_research", forbidden_collection)
    monkeypatch.setattr(lf.research, "complete", complete)
    result = await lf.run_long_report(
        {"model_id": "fixture", "analyses": ["properties"], "lang": "en"},
        lambda state: None, checkpoint=checkpoint)
    assert checkpoint == saved
    assert result["completed_chapters"] == 8 and result["model_calls"] == 17
    assert len(calls) == 2 and result["chapters"][6]["attempts"] == 3
    assert result["partial"] is (collection_partial is not False)
    assert result["review_status"] == "completed"
    for i in range(8):
        if i != 6:
            assert all(result["chapters"][i][key] == value for key, value in saved["chapters"][i].items())
