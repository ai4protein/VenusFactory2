"""
tools/train/tools_mcp.py — FastMCP wrappers for local model training + registry.

Mounted by src/mcp_server.py. Online mode refuses these tools (heavy GPU compute).
Science Agent (kimi) discovers them via MCP introspection allowlist.
"""
from __future__ import annotations

from typing import List, Optional

from fastmcp import FastMCP

from tools.runtime_tool_policy import assert_tool_allowed, tool_denied_json
from tools.train.model_registry import list_trained_models_json, register_trained_model_json
from tools.train.train_operations import (
    process_csv_and_generate_config,
    run_predict_tool,
    run_train_tool,
)

mcp = FastMCP("Venus_Train_MCP")


def _guarded(tool_name: str, fn, *args, **kwargs) -> str:
    try:
        assert_tool_allowed(tool_name)
    except RuntimeError:
        return tool_denied_json(tool_name)
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        import json
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)


@mcp.tool(name="generate_training_config")
def mcp_generate_training_config(
    csv_file: Optional[str] = None,
    dataset_path: Optional[str] = None,
    valid_csv_file: Optional[str] = None,
    test_csv_file: Optional[str] = None,
    output_name: str = "custom_training_config",
    user_requirements: Optional[str] = None,
) -> str:
    """Generate training JSON from CSV or Hugging Face dataset (local mode only)."""
    return _guarded(
        "generate_training_config",
        process_csv_and_generate_config,
        csv_file,
        valid_csv_file,
        test_csv_file,
        output_name,
        dataset_path=dataset_path,
        user_requirements=user_requirements,
    )


@mcp.tool(name="train_protein_model")
def mcp_train_protein_model(config_path: str) -> str:
    """Train a protein model from a config JSON. Auto-registers under ckpt/user_trained (local only)."""
    return _guarded("train_protein_model", run_train_tool, config_path)


@mcp.tool(name="protein_model_predict")
def mcp_protein_model_predict(
    config_path: str,
    sequence: Optional[str] = None,
    csv_file: Optional[str] = None,
) -> str:
    """Predict with a trained/registered model. ``config_path`` may be a file path or model_id (local only)."""
    return _guarded(
        "protein_model_predict",
        run_predict_tool,
        config_path,
        sequence=sequence,
        csv_file=csv_file,
    )


@mcp.tool(name="register_trained_model")
def mcp_register_trained_model(
    config_path: str,
    model_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
) -> str:
    """Register an existing checkpoint into ckpt/user_trained for cross-session reuse (local only)."""
    return _guarded(
        "register_trained_model",
        register_trained_model_json,
        config_path=config_path,
        model_id=model_id,
        output_dir=output_dir,
        model_path=model_path,
    )


@mcp.tool(name="list_trained_models")
def mcp_list_trained_models() -> str:
    """List models registered under ckpt/user_trained (local only)."""
    return _guarded("list_trained_models", list_trained_models_json)
