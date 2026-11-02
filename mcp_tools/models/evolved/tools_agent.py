"""Local chat tools for models published by persistent Scientist goals."""

import json

from langchain.tools import tool
from langchain_core.runnables import RunnableConfig

from mcp_tools.models.evolved.service import dispatch


@tool("list_evolved_models")
def list_evolved_models_tool(query: str = "", stage: str = "", limit: int = 10, offset: int = 0) -> str:
    """Search Scientist model publications by text or engineering stage. Results are revalidated on use. Local only."""
    return json.dumps(dispatch("list", query=query, stage=stage, limit=limit, offset=offset), ensure_ascii=False)


@tool("describe_evolved_model")
def describe_evolved_model_tool(entry_id: str) -> str:
    """Revalidate a published Scientist model and read its exact scope, validation metrics and required inputs. Local only."""
    return json.dumps(dispatch("describe", entry_id=entry_id), ensure_ascii=False)


@tool("predict_with_evolved_model")
def predict_with_evolved_model_tool(entry_id: str, inputs: list[dict], scope: dict, config: RunnableConfig) -> str:
    """Predict with a verified Scientist model. Copy the exact scope from describe and follow its input_fields and input_contract for 1-256 records. Context models require id/protein/group/values; sequence models require id/sequence (features when specified). No target labels, training or publication required. Local only."""
    host = config.get("configurable", {})
    session = host.get("session_id") or host.get("thread_id")
    return json.dumps(
        dispatch(
            "predict", entry_id=entry_id, inputs=inputs, scope=scope, session_id=session, session_binding="chat_session"
        ),
        ensure_ascii=False,
    )


EVOLVED_MODEL_TOOLS = [list_evolved_models_tool, describe_evolved_model_tool, predict_with_evolved_model_tool]
