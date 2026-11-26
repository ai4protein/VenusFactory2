import json
from backend.web_v2.chat_history import include_research_content
from backend.web_v2.science_goals import research_live

def test_history_prefers_running_goal_and_does_not_spin_for_waiting_or_dead_worker(tmp_path):
    ids=["a"*32,"b"*32]
    for i, gid in enumerate(ids):
        (tmp_path/(gid+".json")).write_text(json.dumps({"created_at":i,"spec":{"objective":"research"}}))
        (tmp_path/gid).mkdir()
        (tmp_path/gid/"goal.json").write_text(json.dumps({"status":"active" if i==0 else "waiting_for_input"}))
    summary={"status":"completed","has_content":True,"title":"Topic"}
    assert include_research_content(summary,tmp_path,lambda _:True)["status"] == "active"
    assert include_research_content(summary,tmp_path,lambda _:False)["status"] == "waiting_for_input"
    (tmp_path/ids[1]/"goal.json").write_text(json.dumps({"status":"completed"}))
    assert include_research_content(summary,tmp_path,lambda _:False)["status"] == "completed"

def test_live_phase_and_checkpoint_are_visible_without_accepting_plan_steps(tmp_path):
    folder=tmp_path/"episodes"/"cycle_000"; (folder/"tools").mkdir(parents=True)
    (folder/"phase.json").write_text(json.dumps({"role":"SC","episode":1}))
    (folder/"tools"/"checkpoint.json").write_text(json.dumps({"summary":"Measured accuracy","completed_steps":["s1"]}))
    state={"cycles":[],"inflight":{"index":0},"progress":{"plan":{"steps":[{"id":"s1","status":"pending"}]}}}
    result=research_live(tmp_path,state)
    assert result["phase"]["role"] == "SC" and result["checkpoint"]["completed_steps"] == ["s1"]
    assert state["progress"]["plan"]["steps"][0]["status"] == "pending"
