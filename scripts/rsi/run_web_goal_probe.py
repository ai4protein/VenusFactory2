"""Start one real GLM Science goal through the isolated web route.

The HTTP session store is isolated; the production service is not involved.
The detached worker must continue after this TestClient and launcher exit.
"""
import asyncio
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research_runs/rsi/20261011_science_web_goal_glm51_v1"
OUT.mkdir(exist_ok=False)
os.environ["VENUSFACTORY_SESSION_DB_PATH"] = str(OUT / "sessions.sqlite")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.harness.contracts import file_hash
from backend.web_v2.chat_api import _shared
from backend.web_v2.chat_api import science_goals as api
from backend.web_v2.chat_api._hooks_runtime import _session_store
from backend.web_v2.science_goals import ScienceGoalService

SID = "web-science-parpgb-glm51-v1"
asyncio.run(_session_store.create({"session_id": SID, "history": [], "created_at": "2026-10-11", "chat_mode": "science_expert"}))
api.service = ScienceGoalService(state_root=OUT / "goals")
api._runtime_mode = lambda: "local"
_shared._runtime_mode = lambda: "local"
app = FastAPI()
app.include_router(api.router, prefix="/api/chat")
payload = {"request_id": "web-real-glm51-parpgb-v1", "driver": "generated_model.v1", "parameters": {"case": "alde_parpgb"}, "objective": "Develop deterministic protein prediction model code for ParPgb, independently verify its training and validation, then reuse the accepted model in a fresh expert-team session to predict the unseen inputs. Continue across episodes until both evidence criteria pass.", "budget": {"max_cycles": 4, "model_calls": 72, "tool_calls": 36, "wall_seconds": 2400}}
with TestClient(app) as client:
    response = client.post(f"/api/chat/sessions/{SID}/science-goals", json=payload)
    (OUT / "launch_response.json").write_text(json.dumps({"http_status": response.status_code, "body": response.json()}, indent=2))
    response.raise_for_status()
    goal = response.json()
    folder, _ = api.service.paths(SID, goal["id"])
    record = {"session_id": SID, "goal_id": goal["id"], "goal_root": str(folder), "transport": "FastAPI TestClient, isolated persistent SQLite session store", "frontend_involved_in_launch": False, "request": payload, "web_implementation": {f: file_hash(ROOT / f) for f in ["backend/web_v2/science_goals.py", "backend/web_v2/chat_api/science_goals.py"]}, "scope": "Web lifecycle integration; no efficacy comparison"}
    (OUT / "protocol.json").write_text(json.dumps(record, indent=2))
    print(json.dumps({"http_status": response.status_code, "goal_id": goal["id"], "worker_alive": goal["worker_alive"], "goal_root": str(folder)}), flush=True)
