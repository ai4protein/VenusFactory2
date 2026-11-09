"""Local chat tools for models published by persistent Scientist goals."""

import json

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from mcp_tools.models.evolved.service import dispatch


@tool("list_evolved_models")
def list_evolved_models_tool(query: str = "", stage: str = "", limit: int = 10, offset: int = 0) -> str:
    """Search Scientist model publications by text or engineering stage. Results are revalidated on use. Local only."""
    return json.dumps(dispatch("list", query=query, stage=stage, limit=limit, offset=offset), ensure_ascii=False)


@tool("describe_evolved_model")
def describe_evolved_model_tool(entry_id: str) -> str:
    """Read a registered Scientist model's exact scope, required inputs and recorded validation metrics. Prediction revalidates current evidence. Local only."""
    return json.dumps(dispatch("describe", entry_id=entry_id), ensure_ascii=False)


@tool("predict_with_evolved_model")
def predict_with_evolved_model_tool(
    entry_id: str, inputs: list[dict], config: RunnableConfig,
    scope_ref: str | None = None, scope: dict | None = None,
) -> str:
    """Predict with a registered Scientist model using scope_ref from describe_evolved_model. Follow its input_fields and input_contract for 1-256 unlabelled records; preserve record IDs and values. The full inline scope is also accepted. Evidence and inputs are revalidated; this never trains or publishes a model. Local only."""
    host = config.get("configurable", {})
    session = host.get("session_id") or host.get("thread_id")
    return json.dumps(
        dispatch(
            "predict", entry_id=entry_id, inputs=inputs, scope=scope, scope_ref=scope_ref,
            session_id=session, session_binding="chat_session"
        ),
        ensure_ascii=False,
    )


EVOLVED_MODEL_TOOLS = [list_evolved_models_tool, describe_evolved_model_tool, predict_with_evolved_model_tool]
