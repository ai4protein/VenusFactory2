"""MCP access to the same owner-configured Scientist model library."""

from fastmcp import Context, FastMCP

from mcp_tools.models.evolved.service import dispatch

mcp = FastMCP("Venus_Evolved_Models")


@mcp.tool(name="list_evolved_models")
def list_evolved_models(query: str = "", stage: str = "", limit: int = 10, offset: int = 0) -> dict:
    """Search recorded Scientist models. Read a selected model's input contract with describe. Local only."""
    return dispatch("list", query=query, stage=stage, limit=limit, offset=offset)


@mcp.tool(name="describe_evolved_model")
def describe_evolved_model(entry_id: str) -> dict:
    """Read a registered Scientist model's exact scope, input fields and recorded validation metrics. Prediction revalidates current evidence. Local only."""
    return dispatch("describe", entry_id=entry_id)


@mcp.tool(name="predict_with_evolved_model")
def predict_with_evolved_model(
    entry_id: str, inputs: list[dict], ctx: Context,
    scope_ref: str | None = None, scope: dict | None = None,
) -> dict:
    """Predict with a registered model using scope_ref from describe_evolved_model and its input_fields/input_contract for 1-256 unlabelled records. Preserve IDs and values. The full inline scope is also accepted. Evidence and inputs are revalidated; this never trains or publishes models. Local only."""
    return dispatch(
        "predict",
        entry_id=entry_id,
        inputs=inputs,
        scope=scope,
        scope_ref=scope_ref,
        session_id=ctx.session_id,
        session_binding="mcp_transport_session",
    )
