"""Evidence-bound result reporting; no new computation or biological claims."""

import hashlib
import json
import os
from pathlib import Path

from fastmcp import FastMCP

from agent.rsi.case_evidence import validate_review

mcp = FastMCP("VenusFactoryEvidenceReview")
BUNDLE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("VENUS_RSI_WORKSPACE", "/workspace"))
ATTEMPTS = 0


@mcp.tool()
def inspect_case_evidence() -> dict:
    """Read controller-derived facts from a pinned domain report. Experimental source rows and predicted outputs are distinct. These facts define the permissible report scope."""
    return json.loads((BUNDLE / "evidence.json").read_text())


@mcp.tool()
def submit_evidence_review(claims_json: str) -> dict:
    """Submit only the nested facts object from inspect_case_evidence, without job_id/plugin/report_sha256 envelope fields. Exactly two attempts are allowed. A rejected claim returns feedback for one correction; rejection is not acceptance. The controller independently validates the saved report."""
    global ATTEMPTS
    if ATTEMPTS >= 2:
        raise ValueError("review attempt budget exhausted")
    if (ROOT / "review.json").exists() or (ROOT / "review.json").is_symlink():
        raise ValueError("review already submitted")
    expected = inspect_case_evidence()["facts"]
    ATTEMPTS += 1
    attempt = {"attempt": ATTEMPTS, "submitted_sha256": hashlib.sha256(claims_json.encode()).hexdigest()}
    try:
        claims = json.loads(claims_json)
        result = validate_review(claims, expected)
    except ValueError as exc:
        feedback = {
            "accepted": False,
            "status": "claims_rejected",
            "reason": str(exc)[:1000],
            "attempts": ATTEMPTS,
            "attempts_remaining": 2 - ATTEMPTS,
            "expected_fields": sorted(expected),
            "instruction": "Submit exactly the facts object; keep all field values/types and omit the metadata envelope.",
        }
        (ROOT / ("review_attempt_" + str(ATTEMPTS) + ".json")).write_text(
            json.dumps({**attempt, "feedback": feedback}, indent=2) + "\n"
        )
        return feedback
    dest = ROOT / "review.json"
    dest.write_text(json.dumps(result, indent=2) + "\n")
    feedback = {
        "accepted": True,
        "artifact": "review.json",
        "claims_sha256": result["claims_sha256"],
        "attempts": ATTEMPTS,
    }
    (ROOT / ("review_attempt_" + str(ATTEMPTS) + ".json")).write_text(
        json.dumps({**attempt, "feedback": feedback}, indent=2) + "\n"
    )
    return feedback


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
