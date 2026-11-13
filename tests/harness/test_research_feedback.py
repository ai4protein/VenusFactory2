import json
import pytest

from agent.harness import Budget, ModelBinding, Profile
from agent.harness.autonomous_research import AutonomousResearch, CRITERIA, execution_feedback
from agent.science_goal import GoalSpec, GoalStore


def failure(name, message, identity="call"):
    return [
        {"kind":"tool_call_start", "tool_call_id":identity,
         "tool_name":"mcp__venusfactory__call_tool", "args":{"name":name,"arguments":{"private_input":identity}}},
        {"kind":"tool_result", "tool_call_id":identity, "is_error":True, "output":message},
    ]


def test_failure_feedback_is_bounded_and_excludes_arguments(tmp_path):
    events=[]
    for n in range(8):
        events.extend(failure("evaluate_research_model", str(n)+"x"*2000, str(n)))
    events.append({"kind":"tool_result", "is_error":False, "output":"successful raw training data"})
    rows=execution_feedback(events,tmp_path)
    assert len(rows)==5
    assert all(len(row["error"])<=1200 for row in rows)
    assert rows[0]["error"].startswith("3")
    assert "private_input" not in json.dumps(rows)
    assert "successful raw training data" not in json.dumps(rows)
    assert execution_feedback([],tmp_path)==[]


@pytest.mark.asyncio
@pytest.mark.parametrize("diagnostic", [
    "Leakage: groups overlap splits",
    "Model source check failed: line 13, column 78: unmatched ']'. Repair the saved source.\n"
    "    counts = [part.count(residue) for residue in AMINO_ACIDS]]\n"
    "                                                         ^",
])
async def test_next_iteration_receives_actual_failures_not_only_critic_summary(tmp_path, diagnostic):
    profile=Profile("science_expert","expert-deepseek-harness",
                    ModelBinding("dmx","unisound-u2-flash","https://www.dmxapi.cn/v1"),
                    Budget(model_calls=12,tool_calls=6,wall_seconds=30))
    store=GoalStore.create(tmp_path/"goal",GoalSpec("Inspect a peptide dataset",CRITERIA,
                           max_cycles=2,model_calls=24,tool_calls=12,wall_seconds=90),profile)
    (store.root/"workspace").mkdir()
    seen=[]
    async def consult(role, context, *args):
        if role=="PI": return "Inspect the actual uploaded data."
        if role=="CB":
            seen.append(context["progress"].get("execution_feedback"))
            return json.dumps({"summary":"Inspect measurements","steps":[
                {"id":"inspect","title":"Inspect data","acceptance":"Actual summary"}],"needs_input":[]})
        if role=="SC":return json.dumps({"decision":"needs_revision","summary":"Unfinished work",
            "issues":["Continue"],"evidence_ids":[],"next_actions":["Correct the execution failure"]})
        raise AssertionError(role)
    async def executor(request,service,events):
        events.extend(failure("evaluate_research_model",diagnostic
            + "\nExecution budget: 4 tool calls remain; 15 read-only discovery calls remain. "
            "Save a checkpoint when the current step is ready. Checkpoint saves have a separate allowance: 2 remain."))
        events.append({"kind":"completed","text":"Partial work","event_audit":{"passed":True}})
    state=await AutonomousResearch(store,"fixture",{},consult=consult,executor=executor,tools=[]).run()
    assert state["status"]=="budget_exhausted"
    assert state["cycles"][0]["summary"] == "Execution needs revision: " + diagnostic.splitlines()[0]
    assert seen[0] is None
    assert seen[1]["errors"]==[{"tool":"evaluate_research_model","error":diagnostic}]
    assert seen[1]["iteration"]==1
    assert state["progress"]["execution_feedback"]["iteration"]==2


def test_successful_retry_clears_only_the_matching_invocation(tmp_path):
    events=failure("evaluate_research_model","Leakage detected","failed")
    other=failure("evaluate_research_model","other error","different")
    other[-1].update(is_error=False,output="Other input passed")
    events.extend(other)
    assert len(execution_feedback(events,tmp_path))==1
    retry=failure("evaluate_research_model","unused","failed")
    retry[-1].update(is_error=False,output="Corrected split passed")
    events.extend(retry)
    assert execution_feedback(events,tmp_path)==[]


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", [False, True])
@pytest.mark.parametrize("completed_steps", [[], ["draft"], ["draft", "review"]])
@pytest.mark.parametrize("critic_ready", [False, True])
async def test_sc_gate_blocks_rejection_without_requiring_completed_review_steps(tmp_path, monkeypatch, tamper, completed_steps, critic_ready):
    from agent.harness import autonomous_research as ar
    profile=Profile("science_expert","expert-deepseek-harness",
                    ModelBinding("dmx","unisound-u2-flash","https://www.dmxapi.cn/v1"),
                    Budget(model_calls=12,tool_calls=6,wall_seconds=30))
    store=GoalStore.create(tmp_path/"goal",GoalSpec("Create a reusable peptide analysis tool",CRITERIA,
                           max_cycles=1,model_calls=12,tool_calls=6,wall_seconds=90),profile)
    (store.root/"workspace").mkdir()
    reviewed=[]
    async def consult(role, context):
        if role=="PI": return "Produce a reusable analysis."
        if role=="CB": return json.dumps({"summary":"Implement and review","steps":[
            {"id":"draft","title":"Prepare candidate","acceptance":"Candidate README"},
            {"id":"review","title":"Complete independent review","acceptance":"Verified review receipt","verification":"review"}],
            "needs_input":[],"deliverable":"tool"})
        if role=="SC": return json.dumps({
            "decision":"ready_for_host_validation" if critic_ready else "needs_revision",
            "summary":"Useful analysis ready for technical Review" if critic_ready else "Result does not meet the research goal",
            "issues":[] if critic_ready else ["PI must revise the scientific approach"]})
        raise AssertionError(role)
    async def executor(request, service, events):
        await service.write_research_file("README.md", "Candidate awaiting independent verification.")
        await service.submit_research_checkpoint("Candidate prepared", completed_steps, ["README.md"], [])
        if tamper: (service.workspace/"README.md").write_text("Changed after checkpoint")
        bundle=store.root/"candidate";bundle.mkdir()
        (bundle/"manifest.json").write_text(json.dumps({"id":"candidate","name":"Candidate"}))
        service.proposals.append(bundle)
        events.append({"kind":"completed","text":"Review is pending","event_audit":{"passed":True}})
    async def review(bundle, **kwargs):
        reviewed.append(bundle.name)
        receipt={"status":"needs_revision","issues":["Add boundary tests"]}
        (bundle/"admission.json").write_text(json.dumps(receipt))
        return receipt
    # This is an offline coordinator test. Transport audit has separate integration coverage.
    monkeypatch.setattr(ar,"audit_requests",lambda *a:{"passed":True,"model_calls":3})
    monkeypatch.setattr(ar,"review_proposal",review)
    state=await AutonomousResearch(store,"fixture",{},consult=consult,executor=executor,tools=[]).run()
    assert reviewed==(["candidate"] if critic_ready and not tamper else [])
    assert state["status"]!="completed"
    assert not state["progress"].get("capabilities")
    if critic_ready and not tamper:
        assert state["progress"]["publication_reviews"][0]["issues"]==["Add boundary tests"]
    elif not critic_ready:
        assert state["progress"]["sc_revision"]["issues"]
        assert not state["progress"].get("publication_reviews")


def test_completed_plan_still_requests_missing_independent_submission():
    from agent.harness.autonomous_research import development_next_action
    summary={"generations":[{"id":"baseline","variant":"baseline","parent":None},
                            {"id":"candidate","variant":"candidate","parent":"baseline"}]}
    step=development_next_action(summary,{"plan":{"steps":[{"status":"completed"}]}})
    assert step["stage"]=="submit_for_review"
    assert step["tool"]=="propose_research_capability"
    assert step["eligible_experiment_ids"]==["candidate"]
    assert "review_spec.development_experiment_id" in step["instruction"]
    assert "approved" not in step


def test_development_guidance_does_not_invent_parent_experiments():
    from agent.harness.autonomous_research import development_next_action
    assert development_next_action({"generations":[]},{})["stage"]=="evaluate_baseline"
    summary={"generations":[{"id":"measured","variant":"baseline","parent":None}]}
    step=development_next_action(summary,{})
    assert step["stage"]=="evaluate_revision"
    assert step["eligible_parent_ids"]==["measured"]
    step=development_next_action(summary,{"publication_reviews":[
        {"status":"needs_revision","issues":["Prediction failed on empty input"]}]})
    assert step["stage"]=="revise_submission"
    assert step["issues"]==["Prediction failed on empty input"]


def test_traceback_summary_keeps_exception_after_stack():
    from agent.harness.research_execution_feedback import failure_summary
    assert failure_summary("Traceback (most recent call last):\n  File \"model.py\", line 2\n    1 / 0\n    ~~~~^\nZeroDivisionError: division by zero") == "ZeroDivisionError: division by zero"
