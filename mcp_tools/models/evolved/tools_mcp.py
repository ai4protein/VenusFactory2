"""MCP access to the same owner-configured Scientist model library."""

from fastmcp import Context, FastMCP

from mcp_tools.models.evolved.service import dispatch

mcp = FastMCP("Venus_Evolved_Models")


@mcp.tool(name="list_evolved_models")
def list_evolved_models(query: str = "", stage: str = "", limit: int = 10, offset: int = 0) -> dict:
    """Search recorded Scientist models. Current evidence is checked by describe and predict. Local only."""
    return dispatch("list", query=query, stage=stage, limit=limit, offset=offset)


@mcp.tool(name="describe_evolved_model")
def describe_evolved_model(entry_id: str) -> dict:
    """Revalidate a Scientist model and return its applicability scope, validation metrics and input fields. Local only."""
    return dispatch("describe", entry_id=entry_id)


@mcp.tool(name="predict_with_evolved_model")
def predict_with_evolved_model(entry_id: str, inputs: list[dict], scope: dict, ctx: Context) -> dict:
    """Predict with a verified model using its exact described scope, input_fields and input_contract for 1-256 records. Context models use id/protein/group/values; sequence models use id/sequence (features when specified). No target labels required. This never trains or publishes models. Local only."""
    return dispatch(
        "predict",
        entry_id=entry_id,
        inputs=inputs,
        scope=scope,
        session_id=ctx.session_id,
        session_binding="mcp_transport_session",
    )
