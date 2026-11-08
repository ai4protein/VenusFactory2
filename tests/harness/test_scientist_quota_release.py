import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from backend.web_v2.science_goals import ScienceGoalService
from backend.web_v2.online_access import AccessStore


def test_scientist_completion_releases_the_public_session_slot(tmp_path, monkeypatch):
    from backend.web_v2 import job_queue, online_access

    sid = "public-scientist-session"
    folder = tmp_path / hashlib.sha256(sid.encode()).hexdigest() / "goal"
    folder.mkdir(parents=True)
    (folder.parent / "goal.json").write_text(json.dumps({"session_id": sid, "online_owner": "visitor"}))
    instance = object.__new__(ScienceGoalService)
    instance.root, instance.children = tmp_path, {}
    instance.live = lambda folder: False
    instance.frozen_profile = lambda folder: SimpleNamespace(harness="expert-deepseek-harness")
    instance.environment = lambda profile: {}
    captured = {}
    def launch(command, **kwargs):
        captured.update(kwargs)
        return "job", SimpleNamespace(pid=123, identity={"pid": 123})
    monkeypatch.setattr(job_queue, "launch", launch)
    monkeypatch.setattr(online_access, "online", lambda: True)

    store = AccessStore(tmp_path / "access.sqlite")
    store.science_slot("chat:" + sid, "ip", "visitor")
    instance.spawn(folder, SimpleNamespace(run_command=lambda *args: ["test-worker"]), "run")
    assert captured["metadata"]["session_id"] == sid
    assert captured["metadata"]["quota_scope"] == captured["session_key"] == "chat:" + sid
    # Same release operation used by both worker implementations, without a UI poll.
    store.release_science(captured["metadata"]["quota_scope"])
    store.science_slot("chat:next-study", "ip", "visitor")
