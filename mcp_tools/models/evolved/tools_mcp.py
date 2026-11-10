"""MCP access to the same owner-configured Scientist model library."""

from fastmcp import Context, FastMCP
from fastmcp.tools import Tool

from mcp_tools.models.evolved.contract import PREDICTION_DESCRIPTION, PredictionArguments
from mcp_tools.models.evolved.service import dispatch
from mcp_tools.scheduler.context import mcp_session_id

mcp = FastMCP("Venus_Evolved_Models")


@mcp.tool(name="list_evolved_models")
def list_evolved_models(query: str = "", stage: str = "", limit: int = 10, offset: int = 0) -> dict:
    """Search recorded Scientist models. Read a selected model's input contract with describe. Local only."""
    return dispatch("list", query=query, stage=stage, limit=limit, offset=offset)


@mcp.tool(name="describe_evolved_model")
def describe_evolved_model(entry_id: str) -> dict:
    """Read a registered Scientist model's exact scope, input fields and recorded validation metrics. Prediction revalidates current evidence. Local only."""
    return dispatch("describe", entry_id=entry_id)


def predict_with_evolved_model(
    entry_id: str, inputs: list[dict], ctx: Context,
    scope_ref: str | None = None, scope: dict | None = None,
) -> dict:
    """Execute the shared prediction contract in the current MCP session."""
    return dispatch(
        "predict",
        entry_id=entry_id,
        inputs=inputs,
        scope=scope,
        scope_ref=scope_ref,
        session_id=mcp_session_id(ctx.session_id),
        session_binding="mcp_transport_session",
    )


# FastMCP infers independent parameters from a function signature. Publish the
# same cross-field contract enforced by the service and Agent queue instead.
mcp.add_tool(Tool.from_function(
    predict_with_evolved_model, description=PREDICTION_DESCRIPTION,
).model_copy(update={"parameters": PredictionArguments.model_json_schema()}))
