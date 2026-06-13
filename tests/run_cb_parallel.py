#!/usr/bin/env python3
"""Parallel e2e test for the 4 SUGGESTED_PROMPTS (chat empty-state cards).

Full multi-stage flow per prompt:
    1) create session
    2) POST /messages/stream  → may produce clarification questions
    3) if clarification:  POST /clarification/respond/stream auto-picking option 0
    4) if plan produced:  POST /plan/confirm/stream with auto_execute=True
    5) drain SSE → final state has plan + tool_executions + status
    6) decide step_decide=continue if waiting_for_step_review (auto_execute should
       skip this but defensively handle)

Usage:
    python tests/run_cb_parallel.py [--base http://127.0.0.1:7861]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

import httpx

# Mirror frontend/src/components/ChatTimeline.tsx:401-422 verbatim.
PROMPTS: list[tuple[str, str]] = [
    (
        "Enzyme Redesign",
        "Search PubMed for recent PETase engineering studies, download PDB 5XJH, "
        "and use ProteinMPNN to redesign the active-site region for improved thermostability",
    ),
    (
        "Mutation Prediction",
        "Download AlphaFold structure for human EGFR (P00533), predict beneficial mutations "
        "with ESM2 and ProtSSN, and identify top 10 stabilizing candidates",
    ),
    (
        "Homolog Survey",
        "Download UniProt sequence for human carbonic anhydrase II (P00918), "
        "find its top homologs via MMseqs2 against UniRef, and compute physicochemical "
        "properties (MW, pI, GRAVY) for each to compare the family",
    ),
    (
        "Functional Analysis",
        "Predict the function of protein P04637, find its interaction partners on STRING, "
        "and check tissue expression from Human Protein Atlas",
    ),
]


async def drain_sse(resp) -> tuple[dict[str, Any] | None, int, int]:
    """Read SSE stream to completion. Return (last_state, event_count, state_event_count)."""
    final: dict[str, Any] | None = None
    event_count = 0
    state_events = 0
    buffer = ""
    done = False
    async for chunk in resp.aiter_text():
        buffer += chunk
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            event = "message"
            data = ""
            for line in block.split("\n"):
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data += line[5:].strip()
            event_count += 1
            if event == "state" and data:
                try:
                    final = json.loads(data)
                    state_events += 1
                except json.JSONDecodeError:
                    pass
            if event == "done":
                done = True
                break
        if done:
            break
    return final, event_count, state_events


async def run_one(client: httpx.AsyncClient, title: str, prompt: str, total_timeout: float) -> dict[str, Any]:
    t0 = time.time()
    sid = "????????"
    headers: dict[str, str] = {}
    stages: list[dict[str, Any]] = []
    final_state: dict[str, Any] | None = None

    try:
        # 1) Create session
        r = await client.post("/api/chat/sessions", json={"model": "deepseek-v4-pro"})
        r.raise_for_status()
        sj = r.json()
        sid = sj["session_id"]
        headers = {"x-session-access-token": sj["session_access_token"]}

        # 2) Send initial message
        t1 = time.time()
        async with client.stream(
            "POST",
            f"/api/chat/sessions/{sid}/messages/stream",
            json={"text": prompt, "model": "deepseek-v4-pro"},
            headers=headers,
            timeout=httpx.Timeout(total_timeout, read=total_timeout, connect=15),
        ) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return _err(title, sid, t0, f"messages/stream HTTP {resp.status_code}: {body[:200].decode(errors='ignore')}")
            final_state, ec, se = await drain_sse(resp)
        stages.append({"stage": "messages", "elapsed": time.time() - t1, "events": ec, "state_events": se,
                       "status": (final_state or {}).get("status", ""), "waiting_for": (final_state or {}).get("waiting_for", "")})

        # 3) If waiting for clarification → auto-answer (pick option 0 of each)
        if final_state and (final_state.get("waiting_for") == "clarification"
                            or final_state.get("status") == "waiting_for_clarification"):
            questions = final_state.get("clarification_questions") or []
            answers = [
                {"question_index": i, "selected_options": [0], "custom_text": ""}
                for i in range(len(questions))
            ]
            t2 = time.time()
            async with client.stream(
                "POST",
                f"/api/chat/sessions/{sid}/clarification/respond/stream",
                json={"answers": answers},
                headers=headers,
                timeout=httpx.Timeout(total_timeout, read=total_timeout, connect=15),
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    return _err(title, sid, t0, f"clarification/respond HTTP {resp.status_code}: {body[:200].decode(errors='ignore')}",
                                stages=stages, partial_state=final_state)
                final_state, ec, se = await drain_sse(resp)
            stages.append({"stage": "clarification", "elapsed": time.time() - t2, "events": ec, "state_events": se,
                           "status": (final_state or {}).get("status", ""), "waiting_for": (final_state or {}).get("waiting_for", ""),
                           "answered": len(answers)})

        # 4) If a plan was produced and waiting for confirmation → auto-confirm with auto_execute
        if final_state and (final_state.get("waiting_for") == "plan_confirmation"
                            or final_state.get("status") == "waiting_for_plan_confirmation"):
            plan = final_state.get("plan") or []
            t3 = time.time()
            async with client.stream(
                "POST",
                f"/api/chat/sessions/{sid}/plan/confirm/stream",
                json={"plan": plan, "auto_execute": True},
                headers=headers,
                timeout=httpx.Timeout(total_timeout, read=total_timeout, connect=15),
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    return _err(title, sid, t0, f"plan/confirm HTTP {resp.status_code}: {body[:200].decode(errors='ignore')}",
                                stages=stages, partial_state=final_state)
                final_state, ec, se = await drain_sse(resp)
            stages.append({"stage": "plan_confirm", "elapsed": time.time() - t3, "events": ec, "state_events": se,
                           "status": (final_state or {}).get("status", ""), "waiting_for": (final_state or {}).get("waiting_for", ""),
                           "plan_sent": len(plan)})

        # 5) Drive the graph through whatever review checkpoints remain.
        #    Strategy: skip PI research sub-reports (they're not the CB plan we want
        #    to test), then any step_review → continue, then iteration → "satisfied".
        max_loops = 30
        loops = 0
        while final_state and loops < max_loops:
            wf = final_state.get("waiting_for", "")
            st = final_state.get("status", "")

            # Plan came after a sub-report review or another path
            if wf == "plan_confirmation" or st == "waiting_for_plan_confirmation":
                plan = final_state.get("plan") or []
                t_c = time.time()
                async with client.stream(
                    "POST",
                    f"/api/chat/sessions/{sid}/plan/confirm/stream",
                    json={"plan": plan, "auto_execute": True},
                    headers=headers,
                    timeout=httpx.Timeout(total_timeout, read=total_timeout, connect=15),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        return _err(title, sid, t0, f"plan/confirm HTTP {resp.status_code}: {body[:200].decode(errors='ignore')}",
                                    stages=stages, partial_state=final_state)
                    final_state, ec, se = await drain_sse(resp)
                stages.append({"stage": "plan_confirm", "elapsed": time.time() - t_c, "events": ec, "state_events": se,
                               "status": (final_state or {}).get("status", ""), "waiting_for": (final_state or {}).get("waiting_for", ""),
                               "plan_sent": len(plan)})
            elif wf == "sub_report_review" or st == "waiting_for_sub_report_review":
                t_s = time.time()
                async with client.stream(
                    "POST",
                    f"/api/chat/sessions/{sid}/sub-report/decide/stream",
                    json={"action": "skip"},  # skip PI sub-reports, jump to plan generation
                    headers=headers,
                    timeout=httpx.Timeout(total_timeout, read=total_timeout, connect=15),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        return _err(title, sid, t0, f"sub-report/decide HTTP {resp.status_code}: {body[:200].decode(errors='ignore')}",
                                    stages=stages, partial_state=final_state)
                    final_state, ec, se = await drain_sse(resp)
                stages.append({"stage": "sub_report_skip", "elapsed": time.time() - t_s, "events": ec, "state_events": se,
                               "status": (final_state or {}).get("status", ""), "waiting_for": (final_state or {}).get("waiting_for", "")})
            elif wf == "step_review" or st == "waiting_for_step_review":
                t_d = time.time()
                async with client.stream(
                    "POST",
                    f"/api/chat/sessions/{sid}/step/decide/stream",
                    json={"action": "continue"},
                    headers=headers,
                    timeout=httpx.Timeout(total_timeout, read=total_timeout, connect=15),
                ) as resp:
                    if resp.status_code != 200:
                        body = await resp.aread()
                        return _err(title, sid, t0, f"step/decide HTTP {resp.status_code}: {body[:200].decode(errors='ignore')}",
                                    stages=stages, partial_state=final_state)
                    final_state, ec, se = await drain_sse(resp)
                stages.append({"stage": "step_decide", "elapsed": time.time() - t_d, "events": ec, "state_events": se,
                               "status": (final_state or {}).get("status", ""), "waiting_for": (final_state or {}).get("waiting_for", "")})
            elif wf == "iteration" or st == "waiting_for_iteration":
                # End of pipeline: just say satisfied
                r = await client.post(
                    f"/api/chat/sessions/{sid}/iteration/decide",
                    json={"action": "satisfied"},
                    headers=headers,
                    timeout=httpx.Timeout(30, read=30, connect=15),
                )
                stages.append({"stage": "iteration_done", "elapsed": 0, "events": 0, "state_events": 0,
                               "status": "completed", "waiting_for": ""})
                # Re-fetch state
                rs = await client.get(f"/api/chat/sessions/{sid}", headers=headers, timeout=30)
                if rs.status_code == 200:
                    final_state = rs.json()
                break
            else:
                # No known checkpoint → done
                break
            loops += 1

    except Exception as exc:
        return _err(title, sid, t0, f"{type(exc).__name__}: {exc}", stages=stages, partial_state=final_state)

    except Exception as exc:
        return _err(title, sid, t0, f"{type(exc).__name__}: {exc}", stages=stages, partial_state=final_state)

    if final_state is None:
        return _err(title, sid, t0, "no state event received", stages=stages)

    plan = final_state.get("plan") or []
    tool_execs = final_state.get("tool_executions") or []
    skipped = final_state.get("skipped_steps") or []
    # tool_executions interleaves PI-research search calls with the actual CB plan
    # tool calls and BOTH use 1-indexed `step` numbers. To get the real plan
    # execution count, filter to executions whose tool_name appears in the plan.
    plan_tools = {p.get("tool_name") for p in plan if p.get("tool_name")}
    plan_execs = [e for e in tool_execs if e.get("tool_name") in plan_tools]
    # Match by tool name first, then by step number among matches.
    executed_step_nums = sorted({e.get("step") for e in plan_execs if isinstance(e.get("step"), int)})
    # Detect per-step real failures: parse executions for explicit success:false /
    # status:error envelopes.
    failed_execs = []
    for e in plan_execs:
        out_str = str(e.get("outputs") or "")
        if '"success": false' in out_str.lower().replace(" ", "") or '"status":"error"' in out_str.lower().replace(" ", ""):
            failed_execs.append({"step": e.get("step"), "tool": e.get("tool_name"), "preview": out_str[:200]})

    return {
        "title": title,
        "session": sid[:8],
        "ok": True,
        "elapsed": time.time() - t0,
        "stages": stages,
        "final_status": final_state.get("status", ""),
        "final_waiting_for": final_state.get("waiting_for", ""),
        "plan_total": len(plan),
        "plan_executed": len(executed_step_nums),
        "executed_step_nums": executed_step_nums,
        "skipped_steps": skipped,
        "failed_step": final_state.get("failed_step"),
        "failed_reason": (str(final_state.get("failed_reason"))[:300] if final_state.get("failed_reason") else None),
        "clarification_count": len(final_state.get("clarification_questions") or []),
        "real_failed_execs": failed_execs,
        "plan_summary": [
            {"step": p.get("step"), "tool": p.get("tool_name"),
             "task": (p.get("task_description") or "")[:90]}
            for p in plan
        ],
        "executions": [
            {"step": e.get("step"), "tool": e.get("tool_name"),
             "out_preview": (str(e.get("outputs", ""))[:160]).replace("\n", " ")}
            for e in tool_execs
        ],
    }


def _err(title, sid, t0, msg, *, stages=None, partial_state=None):
    return {
        "title": title, "session": sid[:8], "ok": False,
        "elapsed": time.time() - t0,
        "error": msg,
        "stages": stages or [],
        "partial_status": (partial_state or {}).get("status", "") if partial_state else "",
        "partial_waiting_for": (partial_state or {}).get("waiting_for", "") if partial_state else "",
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:7861")
    ap.add_argument("--per-prompt-timeout", type=float, default=1500)
    args = ap.parse_args()

    print(f"=== Parallel CB e2e: {len(PROMPTS)} prompts → {args.base} ===\n", flush=True)
    async with httpx.AsyncClient(base_url=args.base, timeout=None) as client:
        try:
            r = await client.get("/api/models", timeout=10)
            r.raise_for_status()
            md = r.json()
            print(f"service ok, default={md['default_model']}, active_gw={md.get('active_gateway')}", flush=True)
        except Exception as e:
            print(f"❌ service unreachable: {e}", file=sys.stderr)
            sys.exit(1)
        try:
            r = await client.get("/api/chat/quota", timeout=10)
            print(f"chat quota: {r.json()}\n", flush=True)
        except Exception:
            print()

        tasks = [run_one(client, t, p, args.per_prompt_timeout) for t, p in PROMPTS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    print("\n" + "=" * 72)
    print("PER-PROMPT RESULTS")
    print("=" * 72)

    runs_ok = runs_failed = runs_no_plan = 0
    total_plan_steps = total_executed = 0

    for r in results:
        if isinstance(r, BaseException):
            print(f"\n❌ uncaught: {type(r).__name__}: {r}")
            runs_failed += 1
            continue

        print(f"\n──── {r['title']}  (sid={r['session']})  elapsed={r['elapsed']:.1f}s")
        for s in r.get("stages", []):
            print(f"    · {s['stage']:14s} {s['elapsed']:>6.1f}s status={s.get('status',''):28s} waiting_for={s.get('waiting_for','')}")

        if not r["ok"]:
            print(f"  ❌ ERROR: {r['error']}")
            print(f"     last status={r.get('partial_status')} waiting_for={r.get('partial_waiting_for')}")
            runs_failed += 1
            continue

        runs_ok += 1
        if r["plan_total"] == 0:
            runs_no_plan += 1
            print(f"  ⚠ final_status={r['final_status']} — CB plan never produced")
            if r["clarification_count"]:
                print(f"    asked {r['clarification_count']} clarification(s)")
            continue

        total_plan_steps += r["plan_total"]
        total_executed += r["plan_executed"]
        status_icon = "✅" if r["final_status"] == "success" else ("❌" if "fail" in r["final_status"] else "⚠️")
        print(f"  {status_icon} final_status={r['final_status']}  plan: {r['plan_total']} steps, executed {r['plan_executed']} {r['executed_step_nums']}")
        if r["skipped_steps"]:
            print(f"     ⏭ skipped: {r['skipped_steps']}")
        if r["failed_step"]:
            print(f"     💥 hard-failed step {r['failed_step']}: {r['failed_reason']}")
        if r.get("real_failed_execs"):
            print(f"     ⚠ executions returning explicit failure ({len(r['real_failed_execs'])}):")
            for fe in r["real_failed_execs"]:
                print(f"        [{fe['step']}] {fe['tool']:35s} {fe['preview']}")
        print(f"     ── plan ──")
        for p in r["plan_summary"]:
            marker = "✓" if p["step"] in r["executed_step_nums"] else "✗"
            print(f"     {marker} [{p['step']}] {p['tool']:35s} {p['task']}")
        print(f"     ── tool executions ──")
        for e in r["executions"]:
            print(f"     [{e['step']}] {e['tool']:35s} → {e['out_preview']}")

    print("\n" + "=" * 72)
    print("AGGREGATE")
    print("=" * 72)
    print(f"runs: ok={runs_ok}, failed/exception={runs_failed}, no-plan={runs_no_plan}")
    if total_plan_steps:
        print(f"plan steps: total={total_plan_steps}, executed={total_executed} "
              f"({total_executed/total_plan_steps*100:.0f}% step-level execution)")


if __name__ == "__main__":
    asyncio.run(main())
