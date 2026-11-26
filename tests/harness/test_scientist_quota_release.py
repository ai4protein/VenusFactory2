"""Historical Scientist runs retain their online slot cleanup during migration."""
import asyncio
from types import SimpleNamespace

from fastapi import Response

from backend.web_v2.chat_api import science_goals
from backend.web_v2.online_access import AccessStore


def test_completed_historical_study_releases_slot_without_resuming(tmp_path, monkeypatch):
    store = AccessStore(tmp_path / "access.sqlite")
    store.science_slot("chat:old", "ip", "visitor")
    async def authenticated(*_):
        return {}
    monkeypatch.setattr(science_goals, "access", authenticated)
    monkeypatch.setattr(science_goals, "_runtime_mode", lambda: "online")
    monkeypatch.setattr(science_goals, "AccessStore", lambda: store)
    goal = {"status": "completed", "worker_alive": False}
    monkeypatch.setattr(science_goals, "service", SimpleNamespace(list=lambda _: [goal]))
    assert asyncio.run(science_goals.list_goals("old", None, Response()))["goals"] == [goal]
    store.science_slot("chat:another", "ip", "visitor")
