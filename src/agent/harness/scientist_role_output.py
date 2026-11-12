"""Typed planning/review responses, never executable scientific actions."""
from __future__ import annotations

import copy
import json
import re
from jsonschema import Draft202012Validator, ValidationError


def object_schema(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def specification(role, protocol, *, model_acceptance=False, study_acceptance=False, study_review_policy=None):
    text = {"type": "string", "minLength": 1, "maxLength": 2000}
    strings = {"type": "array", "items": {"type": "string"}}
    if role == "PI":
        name = "return_research_direction"
        schema = object_schema({"direction": {**text, "maxLength": 8000}})
    elif role == "CB":
        name = "return_research_plan"
        schema = object_schema({
            "summary": text,
            "deliverable": {"enum": ["model", "tool", "analysis"]},
            "steps": {"type": "array", "minItems": 1, "maxItems": 24,
                      "items": object_schema({key: text for key in ("id", "title", "acceptance")})},
            "needs_input": strings,
        })
    elif role == "SC":
        name = "return_critic_review"
        schema = object_schema({
            "decision": {"enum": ["ready_for_host_validation", "needs_revision"]},
            "summary": text, "issues": strings,
        })
        if model_acceptance:
            from .scientific_acceptance import model_schema
            schema = model_schema()
        elif study_acceptance:
            schema["properties"]["evidence_ids"] = {
                "type": "array", "minItems": 1, "maxItems": 12, "uniqueItems": True,
                "items": {"type": "string", "pattern": "^[a-f0-9]{32}$"},
                "description": "Host experiment IDs supporting this study review. Cite measurements from development_measurements; these are not accepted model IDs."}
            schema["required"].append("evidence_ids")
            if study_review_policy is not None:
                from .study_review_state import POLICY, request_schema
                if study_review_policy != POLICY:
                    raise ValueError("Unsupported study review evidence policy")
                schema["properties"]["revision_requests"] = request_schema()
                schema["required"].append("revision_requests")
    else:
        return None
    if role == "CB":
        schema["required"].remove("deliverable")
        from .research_verification_contract import VERIFICATION_DESCRIPTION, VERIFICATION_KINDS
        schema["properties"]["steps"]["items"]["properties"]["verification"] = {
            "enum": list(VERIFICATION_KINDS), "description": VERIFICATION_DESCRIPTION,
        }
        schema["properties"]["steps"]["items"]["properties"]["approach"] = {
            **text, "description": (
                "Concrete implementation method. For model training or revision, name the feature representation, "
                "learning procedure, comparison and scientific rationale. Revise this method from measured feedback "
                "while preserving the acceptance criteria."
            ),
        }
    function = {"name": name, "description": "Return this phase's structured output. No action is executed.",
                "parameters": schema}
    if protocol == "chat_completions":
        tool = {"type": "function", "function": function}
        # With one declared output function, required selects the same interface.
        choice = "required"
    elif protocol == "responses":
        tool = {"type": "function", **function}
        choice = {"type": "function", "name": name}
    else:
        raise ValueError("Unsupported scientific role protocol")
    return {"role": role, "protocol": protocol, "name": name, "schema": schema,
            "tool": tool, "choice": choice}


def request_fields(spec):
    fields = {"tools": [copy.deepcopy(spec["tool"])],
              "tool_choice": copy.deepcopy(spec["choice"]), "stream": False}
    # Many Chat Completions providers do not expose this optional parameter.
    # output_content still requires exactly one call and never dispatches it.
    if spec["protocol"] == "responses":
        fields["parallel_tool_calls"] = False
    return fields


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate scientific role field")
        value[key] = item
    return value


def _finite(value):
    raise ValueError("Nonfinite scientific role value")


def output_content(value, spec, model):
    """Extract one exact output channel; never dispatch a returned function."""
    if not isinstance(value, dict) or value.get("model") != model or value.get("error"):
        raise ValueError("Scientific role response identity mismatch")
    # Some compatible endpoints ignore forced output selection and return
    # complete text instead. It must satisfy the same schema and host review.
    text_only = False
    if spec["protocol"] == "chat_completions":
        choices = value.get("choices")
        if isinstance(choices, list) and len(choices) == 1:
            message = choices[0].get("message") or {}
            text_only = (choices[0].get("finish_reason") == "stop"
                         and message.get("role", "assistant") == "assistant"
                         and not message.get("tool_calls") and not message.get("function_call"))
    else:
        output = value.get("output")
        text_only = (isinstance(output, list) and bool(output)
                     and all(isinstance(item, dict) and item.get("type") in {"message", "reasoning"}
                             for item in output))
    if text_only:
        if spec["protocol"] == "chat_completions":
            content = message.get("content")
            if not message.get("refusal") and (content is None or (
                    isinstance(content, str) and len(content) <= 128000 and not content.strip())):
                raise RoleOutputError(
                    "The model completed without an answer. Return the required structured output.", content)
        from .native_expert_team import role_content
        raw = role_content(value, spec["protocol"], model).strip()
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[0] in {"\x60\x60\x60json", "\x60\x60\x60"} and lines[-1] == "\x60\x60\x60":
            raw = '\n'.join(lines[1:-1])
        return validate_output(raw, spec)
    if spec["protocol"] == "chat_completions":
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("Expected one scientific role response")
        choice = choices[0]
        message = choice.get("message") or {}
        calls = message.get("tool_calls")
        if (choice.get("finish_reason") != "tool_calls"
                or message.get("role", "assistant") != "assistant"
                or message.get("function_call")
                or message.get("content") not in (None, "")
                or not isinstance(calls, list) or len(calls) != 1
                or calls[0].get("type") != "function"):
            raise ValueError("Scientific role must return only its structured output")
        function = calls[0].get("function") or {}
    else:
        if value.get("status") != "completed" or value.get("incomplete_details"):
            raise ValueError("Incomplete scientific role response")
        output = value.get("output")
        if not isinstance(output, list) or any(not isinstance(item, dict) for item in output):
            raise ValueError("Invalid scientific role output")
        calls = [item for item in output if item.get("type") != "reasoning"]
        if (len(calls) != 1 or calls[0].get("type") != "function_call"
                or calls[0].get("status") != "completed" or calls[0].get("namespace")):
            raise ValueError("Scientific role must return only its structured output")
        function = calls[0]
    if function.get("name") != spec["name"]:
        raise ValueError("Unexpected scientific role output function")
    raw = function.get("arguments")
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 128000:
        raise ValueError("Unbounded scientific role output")
    return validate_output(raw, spec)


class RoleOutputError(ValueError):
    """A public role response can be repaired without weakening the contract."""
    def __init__(self, message, raw):
        super().__init__(message)
        self.raw = raw[:12000] if isinstance(raw,str) else ""

def validate_output(raw, spec):
    try:
        parsed = json.loads(raw, object_pairs_hook=_unique, parse_constant=_finite)
    except (ValueError, TypeError) as exc:
        raise RoleOutputError("Return one complete JSON object with unique fields and finite values.",raw) from exc
    normalized = False
    # Some compatible models encode the single return function as a JSON envelope.
    # It is output data only; require its exact identity and validate the inner schema.
    if isinstance(parsed, dict) and set(parsed) == {spec["name"]} and isinstance(parsed[spec["name"]], dict):
        parsed = parsed[spec["name"]]
        normalized = True
    if spec["role"] == "CB" and isinstance(parsed,dict) and isinstance(parsed.get("steps"),list):
        for step in parsed["steps"]:
            if isinstance(step,dict):
                for key in ("status","evidence"):
                    if key in step:
                        step.pop(key);normalized = True
    try:
        Draft202012Validator(spec["schema"]).validate(parsed)
    except ValidationError as exc:
        path = ".".join(re.sub(r"[^a-zA-Z0-9_]", "", str(part))[:40] for part in exc.absolute_path)[:160] or "plan"
        detail = "Expected " + str(exc.schema.get("type","the declared fields"))
        if exc.validator == "enum":
            detail = "Allowed values: " + ", ".join(str(x) for x in exc.schema["enum"])
        elif exc.validator in {"required","additionalProperties"}:
            detail = "Allowed fields: " + ", ".join(exc.schema.get("properties",{}))
        raise RoleOutputError("Scientific role schema mismatch at " + path + ". " + detail,raw) from exc
    return json.dumps(parsed,ensure_ascii=False) if normalized else raw
