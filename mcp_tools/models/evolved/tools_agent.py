"""Local chat tools for models published by persistent Scientist goals."""

import json

from langchain_core.tools import StructuredTool, tool
from langchain_core.runnables import RunnableConfig

from mcp_tools.models.evolved.contract import PREDICTION_DESCRIPTION, PredictionArguments
from mcp_tools.models.evolved.service import dispatch


@tool("list_evolved_models")
def list_evolved_models_tool(query: str = "", stage: str = "", limit: int = 10, offset: int = 0) -> str:
    """Search Scientist model publications by text or engineering stage. Results are revalidated on use. Local only."""
    return json.dumps(dispatch("list", query=query, stage=stage, limit=limit, offset=offset), ensure_ascii=False)


@tool("describe_evolved_model")
def describe_evolved_model_tool(entry_id: str) -> str:
    """Read a registered Scientist model's exact scope, required inputs and recorded validation metrics. Prediction revalidates current evidence. Local only."""
    return json.dumps(dispatch("describe", entry_id=entry_id), ensure_ascii=False)


def predict_with_evolved_model_tool(
    entry_id: str, inputs: list[dict], config: RunnableConfig,
    scope_ref: str | None = None, scope: dict | None = None,
) -> str:
    """Execute the shared model prediction contract in the current chat session."""
    host = config.get("configurable", {})
    session = host.get("session_id") or host.get("thread_id")
    return json.dumps(
        dispatch(
            "predict", entry_id=entry_id, inputs=inputs, scope=scope, scope_ref=scope_ref,
            session_id=session, session_binding="chat_session"
        ),
        ensure_ascii=False,
    )


class _PredictionTool(StructuredTool):
    @property
    def tool_call_schema(self):
        # This explicit public contract has no injected fields. LangChain's
        # inferred subset otherwise discards its root-level alternative rules.
        return self.get_input_schema()


predict_with_evolved_model_tool = _PredictionTool.from_function(
    predict_with_evolved_model_tool, name="predict_with_evolved_model",
    description=PREDICTION_DESCRIPTION, args_schema=PredictionArguments,
)
EVOLVED_MODEL_TOOLS = [list_evolved_models_tool, describe_evolved_model_tool, predict_with_evolved_model_tool]
