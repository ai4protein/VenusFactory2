"""Artifact-grounded tool contracts shared by Scientist and Agent runtimes."""

import json

from .paper_step import digest
from .paper_step_tool_service import PaperStepTools


def validate_paper_step_result(name, raw, *, workspace, capability):
    value = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict):
        raise ValueError("Paper step result must be an object")
    value = dict(value)
    if "success" in value and value.pop("success") is not True:
        raise ValueError("Paper step operation failed")
    service = PaperStepTools(workspace, capability)
    if name == "inspect_paper_step_task":
        expected = service.inspect()
    elif name == "discover_paper_steps":
        expected = service.discover()
    elif name == "evaluate_paper_step":
        expected = service.evaluation_receipt(value.get("attempt"))
    elif name == "submit_paper_step":
        expected = service.submission_receipt()
    elif name == "execute_registered_paper_step":
        expected = service.reuse_receipt()
    else:
        raise ValueError("Unknown paper step tool")
    if digest(value) != digest(expected):
        raise ValueError("Paper step tool output differs from its bound artifacts")
