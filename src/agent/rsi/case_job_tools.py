"""MCP request interface for host-controlled asynchronous reproduction jobs."""

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("VenusFactoryReproductions")
ROOT = Path(os.environ.get("VENUS_RSI_WORKSPACE", "/workspace"))


async def request(value):
    rpc = ROOT / "case_rpc"
    rid = uuid.uuid4().hex
    temporary = rpc / (rid + ".tmp")
    temporary.write_text(json.dumps(value))
    temporary.rename(rpc / (rid + ".request.json"))
    response = rpc / (rid + ".response.json")
    async with asyncio.timeout(15):
        while not response.exists():
            await asyncio.sleep(0.1)
    if response.is_symlink():
        raise ValueError("response symlink rejected")
    result = json.loads(response.read_text())
    if not result["success"]:
        raise ValueError(result.get("error", "controller rejected request"))
    return result


@mcp.tool()
def inspect_registered_reproduction() -> dict:
    """Read a host-verified, immutable reproduction recipe from a prior session. This is a reusable workflow, not a newly invented model."""
    path = Path(__file__).resolve().parent / "registered_recipe.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("No registered reproduction recipe is admitted")
    return json.loads(path.read_text())


@mcp.tool()
async def discover_reproduction_plugins() -> dict:
    """Inspect available protein-study workflows, exact scope, budget and submitted jobs."""
    return await request({"action": "catalog"})


@mcp.tool()
async def submit_reproduction(
    plugin_id: str,
    parameters_json: str,
    idempotency_key: Annotated[str, Field(min_length=1, max_length=256, pattern=r"^[a-zA-Z0-9_-]+$")],
) -> dict:
    """Request a pinned workflow at its declared computational scope. Parameters are a JSON object. Use one stable idempotency key of 1..256 ASCII letters, digits, underscores or hyphens. A registered version hash may be included; keep the whole key unchanged when retrying. Returns a durable job identity; queued is not completed."""
    result = await request(
        {
            "action": "submit",
            "plugin_id": plugin_id,
            "parameters": json.loads(parameters_json),
            "idempotency_key": idempotency_key,
        }
    )
    receipt = {
        "schema": 1,
        "job_id": result["job_id"],
        "plugin_id": result["plugin_id"],
        "parameters": result["parameters"],
        "submission_only": True,
    }
    path = ROOT / "case_submission.json"
    if path.is_symlink():
        raise ValueError("Submission receipt symlink rejected")
    if path.exists():
        if json.loads(path.read_text()) != receipt:
            raise ValueError("Submission receipt is already bound to another job")
    else:
        with path.open("x") as stream:
            json.dump(receipt, stream, allow_nan=False)
    return {**result, "submission_receipt": path.name}


@mcp.tool()
async def inspect_reproduction_job(job_id: str) -> dict:
    """Read controller-verified status and domain evidence. Do not infer completion from successful submission."""
    return await request({"action": "status", "job_id": job_id})


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
