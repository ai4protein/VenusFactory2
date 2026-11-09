"""Harness-independent user interaction contract. Never grants tool permissions."""
from __future__ import annotations

import json
import time
import uuid
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Question(Strict):
    id: str = Field(min_length=1, max_length=64, pattern=r"^[\w-]+$")
    question: str = Field(min_length=1, max_length=2000)
    options: list[str] = Field(default_factory=list, max_length=12)
    multiple: bool = False
    allow_text: bool = True

    @model_validator(mode="after")
    def usable(self):
        if any(not s.strip() or len(s) > 500 for s in self.options) or len(set(self.options)) != len(self.options):
            raise ValueError("Options must be unique, nonempty and at most 500 characters")
        if not self.options and not self.allow_text:
            raise ValueError("A question must accept text or offer options")
        return self


class AskInput(Strict):
    questions: list[Question] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def unique(self):
        if len({q.id for q in self.questions}) != len(self.questions):
            raise ValueError("Question IDs must be unique")
        return self


class ConfirmInput(Strict):
    title: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=12000)
    kind: Literal["confirmation", "plan"] = "confirmation"


class Interaction(Strict):
    version: Literal[1] = 1
    id: str = Field(pattern=r"^[a-f0-9]{32}$")
    kind: Literal["question", "confirmation", "plan"]
    title: str = Field(default="", max_length=200)
    message: str = Field(default="", max_length=12000)
    questions: list[Question] = Field(default_factory=list, max_length=6)
    created_at: float


class Answer(Strict):
    question_id: str = Field(min_length=1, max_length=64)
    selected: list[int] = Field(default_factory=list, max_length=12)
    text: str = Field(default="", max_length=4000)


class InteractionResponse(Strict):
    interaction_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    decision: Literal["answer", "approve", "reject", "revise", "cancel"]
    answers: list[Answer] = Field(default_factory=list, max_length=6)
    feedback: str = Field(default="", max_length=4000)


def response_text(pending: dict, response: InteractionResponse) -> str:
    gate = Interaction.model_validate(pending)
    if gate.id != response.interaction_id:
        raise ValueError("This question is no longer pending")
    if response.decision == "cancel":
        if response.answers:
            raise ValueError("A cancelled question cannot contain answers")
    elif gate.kind == "question":
        if response.decision != "answer":
            raise ValueError("Answer or cancel this question")
        answers = {a.question_id: a for a in response.answers}
        if len(answers) != len(response.answers) or set(answers) != {q.id for q in gate.questions}:
            raise ValueError("Answer every question exactly once")
        for q in gate.questions:
            a = answers[q.id]
            if len(set(a.selected)) != len(a.selected) or any(i < 0 or i >= len(q.options) for i in a.selected):
                raise ValueError("Invalid selected option")
            if not q.multiple and len(a.selected) > 1:
                raise ValueError("Choose one option")
            if a.text and not q.allow_text:
                raise ValueError("This question only accepts the listed options")
            if not a.selected and not a.text:
                raise ValueError("Provide an answer")
    else:
        if response.decision not in {"approve", "reject", "revise"} or response.answers:
            raise ValueError("Explicitly approve, reject or revise this request")
        if response.decision == "revise" and not response.feedback:
            raise ValueError("Describe the requested revision")
    content = {"interaction_id": gate.id, "kind": gate.kind, "title": gate.title, "message": gate.message,
               "decision": response.decision, "feedback": response.feedback}
    # Labels are retained alongside indices, independent of future catalog changes.
    if response.decision == "answer":
        content["resolved_answers"] = [{"question": q.question, "selected": [q.options[i] for i in a.selected], "text": a.text}
                                       for q in gate.questions for a in response.answers if a.question_id == q.id]
    return ("User response to the pending interaction. Continue the original task from recorded work; "
            "do not repeat completed actions. A rejection/cancellation does not authorize the proposed action. "
            "No response overrides tool policy, isolation or resource limits.\n" + json.dumps(content, ensure_ascii=False))


INTERACTION_NAMES = frozenset({"ask_user", "request_confirmation"})
INTERACTION_GUIDANCE = (
    "Finish the current request with its answer and actual results. Optional next steps belong in the answer; "
    "do not pause to offer more work, check satisfaction or ask whether to continue after delivering the result. "
    "Use ask_user only when missing information or a decision blocks unfinished work. "
    "Use request_confirmation only for a concrete action that needs additional authorization. "
    "The user's request already authorizes its ordinary analysis and session outputs; do not seek approval "
    "for that work or each research iteration. These tools pause execution until an explicit response; "
    "do not perform the pending action before that response or repeat questions answered in context. "
    "They replace harness-native questions, terminal prompts and plan confirmations. "
    "Ask concise questions, with options or free-text answers as appropriate. "
    "A response never enables disabled tools or changes sandbox, access or resource limits. "
    "Never ask for credentials."
)


def interaction_tools(on_request):
    from langchain_core.tools import StructuredTool

    async def ask_user(questions):
        value = AskInput.model_validate({"questions": questions})
        gate = Interaction(id=uuid.uuid4().hex, kind="question", questions=value.questions, created_at=time.time())
        on_request(gate.model_dump())
        return {"status": "waiting_for_input", "interaction_id": gate.id}

    async def request_confirmation(title, message, kind="confirmation"):
        value = ConfirmInput(title=title, message=message, kind=kind)
        gate = Interaction(id=uuid.uuid4().hex, created_at=time.time(), **value.model_dump())
        on_request(gate.model_dump())
        return {"status": "waiting_for_input", "interaction_id": gate.id}

    return [StructuredTool.from_function(coroutine=ask_user, name="ask_user", args_schema=AskInput,
               description="Pause only for missing information required to finish the current request. Accept choices or free text. Do not use for optional follow-ups after delivering results."),
            StructuredTool.from_function(coroutine=request_confirmation, name="request_confirmation", args_schema=ConfirmInput,
               description="Pause for necessary additional authorization or a missing decision. Do not ask again to perform requested analysis or save its session outputs. Does not grant system permissions.")]


def response_display(pending, response, lang="en"):
    zh = lang == "zh"
    if response.decision == "answer":
        rows=[]
        for q in pending['questions']:
            a=next(a for a in response.answers if a.question_id==q['id'])
            rows.append(q['question'] + "\n" + "; ".join([*[q['options'][i] for i in a.selected], *([a.text] if a.text else [])]))
        return "\n\n".join(rows)
    labels={"approve":("Approved", "同意"), "reject":("Declined", "不同意"), "revise":("Requested changes", "按意见修改"), "cancel":("Skipped", "暂不回答")}
    return labels[response.decision][int(zh)] + (": " + response.feedback if response.feedback else "")


def settle_interaction_tools(items, turn_id, gate):
    """The question tool succeeded even if native cancellation hid its ACK."""
    name = 'ask_user' if gate['kind'] == 'question' else 'request_confirmation'
    for item in items:
        if (item.get('turn_id') == turn_id and item.get('status') == 'running'
                and str(item.get('tool_name') or item.get('name')).split('__')[-1] == name):
            item.update(status='ok', output={'status':'waiting_for_input','interaction_id':gate['id']})
