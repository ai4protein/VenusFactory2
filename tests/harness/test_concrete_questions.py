import json
from types import SimpleNamespace

import pytest

from agent.harness import Profile
from agent.harness.autonomous_research import AutonomousResearch, parse_plan
from agent.harness.research_workspace import ResearchWorkspace

PLACEHOLDERS = [
    "specific missing user information",
    "Specific missing user information, such as the target column for CHS soluble expression.",
    "...",
    "   ",
]

def plan(questions):
    return {"summary": "Inspect provided data", "steps": [
        {"id": "inspect", "title": "Inspect measurements", "acceptance": "Identify measured target"}
    ], "needs_input": questions}

@pytest.mark.parametrize("question", PLACEHOLDERS)
def test_planning_placeholder_is_not_a_user_input_request(question):
    with pytest.raises(ValueError, match="concrete"):
        parse_plan(json.dumps(plan([question])))

@pytest.mark.asyncio
@pytest.mark.parametrize("question", PLACEHOLDERS)
async def test_checkpoint_placeholder_cannot_create_a_pause(tmp_path, question):
    service = ResearchWorkspace(tmp_path / "workspace", tmp_path / "control", parse_plan(json.dumps(plan([]))))
    with pytest.raises(ValueError, match="concrete"):
        await service.submit_research_checkpoint("Fitting needs code repair", [], [], [question])
    assert service.checkpoint is None
    assert not (tmp_path / "control/checkpoint.json").exists()

@pytest.mark.parametrize("question", [
    "Please provide the target protein sequence.",
    "Which measured column should be predicted: activity or expression?",
    "请提供样本分组对应的患者编号。",
])
@pytest.mark.asyncio
async def test_genuine_missing_input_is_preserved_in_plan_and_checkpoint(tmp_path, question):
    parsed = parse_plan(json.dumps(plan([question])))
    assert parsed["needs_input"] == [question]
    service = ResearchWorkspace(tmp_path / "workspace", tmp_path / "control", parsed)
    await service.submit_research_checkpoint("The supplied files lack a required input", [], [], [question])
    assert service.checkpoint["needs_input"] == [question]

@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_question", [PLACEHOLDERS[1], "evaluation_file"])
async def test_existing_bounded_plan_repair_corrects_invalid_question_without_answering_for_user(tmp_path, invalid_question):
    seen = []
    async def consult(role, context):
        seen.append(context)
        return json.dumps(plan([invalid_question] if len(seen) == 1 else []))
    profile = Profile.from_dict({
        "mode": "science_expert", "harness": "expert-deepseek-harness",
        "model": {"provider": "dmx", "model": "glm-4-flash", "base_url": "https://www.dmxapi.cn/v1"},
        "budget": {"model_calls": 12, "tool_calls": 12, "wall_seconds": 120, "output_tokens": 4096}
    })
    runner = AutonomousResearch(None, "fixture", {}, consult=consult, tools=[])
    result = await runner.structured("CB", {"objective": "Study the supplied measurements"},
                                    profile, SimpleNamespace(calls=0), tmp_path, parse_plan)
    assert result["needs_input"] == []
    assert len(seen) == 2
    assert "concrete" in seen[1]["format_repair"]["validation_error"]
    assert seen[1]["format_repair"]["previous_response"] == json.dumps(plan([invalid_question]))
    assert result["steps"][0]["status"] == "pending"


INTERNAL_FIELDS = [
    "evaluation_file", "Evaluation File", " source_file ", "parent_experiment_id",
    "prediction_key", "review_spec", "reproduction_script", "fitted_files",
    chr(96) + "evaluation_file" + chr(96),
    '"evaluation_file"?',
]


@pytest.mark.parametrize("question", INTERNAL_FIELDS)
def test_protocol_field_is_not_a_missing_user_input(question):
    with pytest.raises(ValueError, match="concrete"):
        parse_plan(json.dumps(plan([question])))


@pytest.mark.asyncio
@pytest.mark.parametrize("question", INTERNAL_FIELDS)
async def test_internal_field_cannot_save_a_checkpoint_that_pauses_work(tmp_path, question):
    service = ResearchWorkspace(tmp_path / "workspace", tmp_path / "control", parse_plan(json.dumps(plan([]))))
    await service.submit_research_checkpoint("Work in progress", [], [], [])
    previous = (service.control / "checkpoint.json").read_bytes()
    with pytest.raises(ValueError, match="concrete"):
        await service.submit_research_checkpoint("Fitting needs code repair", [], [], [question])
    assert service.checkpoint["needs_input"] == []
    assert (service.control / "checkpoint.json").read_bytes() == previous


def test_missing_internal_evaluation_reference_explains_how_to_recover():
    with pytest.raises(ValueError) as caught:
        parse_plan(json.dumps(plan(["evaluation_file"])))
    assert "prepare_research_dataset" in str(caught.value)
    assert "prepared_dataset" in str(caught.value)


@pytest.mark.parametrize("question", [
    "Can you provide independent experimental measurements for evaluation_file?",
    "Which mutation assay should be used to evaluate the model?",
    "Please upload the missing source_file named measurements.csv.",
    "请提供独立实验数据作为评估集。",
])
def test_concrete_user_data_questions_can_mention_protocol_fields(question):
    assert parse_plan(json.dumps(plan([question])))["needs_input"] == [question]
