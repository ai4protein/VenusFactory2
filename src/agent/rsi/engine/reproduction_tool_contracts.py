"""Operational validation of immutable reproduction lookups and review receipts.

These contracts do not establish scientific acceptance. The host still checks
the complete tool chain, job artifacts and domain evidence independently.
"""

import json

from .case_evidence import validate_review
from .protein_model import digest
from .tool_contracts import _equal, _read

SUPPORTED = frozenset(
    {
        "inspect_registered_reproduction",
        "inspect_case_evidence",
        "submit_evidence_review",
        "discover_reproduction_plugins",
        "submit_reproduction",
        "inspect_reproduction_job",
        "inspect_paper_step_task",
        "evaluate_paper_step",
        "submit_paper_step",
        "discover_paper_steps",
        "execute_registered_paper_step",
    }
)


def validate_reproduction_result(name, raw, *, workspace, capability):
    if name in {
        "inspect_paper_step_task",
        "evaluate_paper_step",
        "submit_paper_step",
        "discover_paper_steps",
        "execute_registered_paper_step",
    }:
        from .paper_step_tool_contracts import validate_paper_step_result

        return validate_paper_step_result(name, raw, workspace=workspace, capability=capability)
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError("Reproduction result must be an object")
    value = dict(value)
    if "success" in value and value.pop("success") is not True:
        raise ValueError("Reproduction operation reported failure")
    digest(value)
    if name in {"discover_reproduction_plugins", "submit_reproduction", "inspect_reproduction_job"}:
        action = {
            "discover_reproduction_plugins": "catalog",
            "submit_reproduction": "submit",
            "inspect_reproduction_job": "status",
        }[name]
        if name == "submit_reproduction":
            if value.pop("submission_receipt", None) != "case_submission.json":
                raise ValueError("Missing job submission receipt")
            _equal(
                _read(workspace, "case_submission.json"),
                {
                    "schema": 1,
                    "job_id": value["job_id"],
                    "plugin_id": value["plugin_id"],
                    "parameters": value["parameters"],
                    "submission_only": True,
                },
            )
        matched = False
        for path in (workspace / "case_rpc").glob("*.response.json"):
            request_name = str(path.relative_to(workspace)).replace(".response.json", ".request.json")
            request = _read(workspace, request_name)
            if request.get("action") != action:
                continue
            response = dict(_read(workspace, str(path.relative_to(workspace))))
            if response.pop("success", None) is True and digest(response) == digest(value):
                matched = True
                break
        if not matched:
            raise ValueError("Tool output does not match a host-broker response")
    elif name == "inspect_registered_reproduction":
        expected = _read(capability, "registered_recipe.json")
        if (
            expected.get("kind") != "verified_reproduction_recipe"
            or not isinstance(expected.get("plugin"), str)
            or not isinstance(expected.get("parameters"), dict)
            or expected.get("novel_model_claim") is not False
        ):
            raise ValueError("Invalid registered workflow metadata")
        _equal(value, expected)
    elif name == "inspect_case_evidence":
        expected = _read(capability, "evidence.json")
        if not isinstance(expected.get("facts"), dict) or not expected["facts"]:
            raise ValueError("Missing source-bound scientific facts")
        _equal(value, expected)
    elif name == "submit_evidence_review":
        number = value.get("attempts")
        if type(number) is not int or number not in {1, 2}:
            raise ValueError("Invalid review attempt")
        attempt = _read(workspace, "review_attempt_" + str(number) + ".json")
        _equal(value, attempt["feedback"])
        if value.get("accepted") is True:
            review = _read(workspace, "review.json")
            checked = validate_review(review["claims"], _read(capability, "evidence.json")["facts"])
            _equal(review, checked)
            _equal(
                value,
                {
                    "accepted": True,
                    "artifact": "review.json",
                    "claims_sha256": checked["claims_sha256"],
                    "attempts": number,
                },
            )
        elif value.get("accepted") is not False or value.get("status") != "claims_rejected":
            raise ValueError("Invalid rejected-review feedback")
    else:
        raise ValueError("No trusted reproduction result contract")
