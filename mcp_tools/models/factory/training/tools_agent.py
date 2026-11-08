# train: LangChain @tool layer; calls core (train_operations), returns JSON.

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field, model_validator

from mcp_tools.tools.runtime_tool_policy import assert_tool_allowed, tool_denied_json

from mcp_tools.models.factory.training.model_registry import list_trained_models_json, register_trained_model_json
from mcp_tools.models.factory.training.train_operations import process_csv_and_generate_config, run_train_tool, run_predict_tool


class CSVTrainingConfigInput(BaseModel):
    csv_file: Optional[str] = Field(
        default=None,
        description="Path to CSV file with training data (protein sequences and labels). Required unless dataset_path is provided. Expects columns such as aa_seq/sequence and label.",
    )
    dataset_path: Optional[str] = Field(
        default=None,
        description="Hugging Face dataset path (e.g. 'username/dataset_name') or local path. Required unless csv_file is provided. Used when training data comes from Hugging Face.",
    )
    valid_csv_file: Optional[str] = Field(
        default=None,
        description="Optional path to validation CSV file. Same column layout as training CSV. If omitted, validation split may be derived from training data.",
    )
    test_csv_file: Optional[str] = Field(
        default=None,
        description="Optional path to test CSV file for final evaluation. Same column layout as training CSV.",
    )
    output_name: str = Field(
        default="custom_training_config",
        description="Base name for the generated config file (timestamp will be appended). Default: custom_training_config.",
    )
    user_requirements: Optional[str] = Field(
        default=None,
        description="Legacy field. Convert requirements into explicit parameters; this tool does not call an LLM.",
    )
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Explicit training overrides selected by the calling agent: plm_model, training_method, num_epochs, learning_rate, batch_size, max_seq_len and other documented training settings. Omitted values use returned defaults.")

    @model_validator(mode="after")
    def require_csv_or_dataset(self):
        if not self.csv_file and not self.dataset_path:
            raise ValueError("Either csv_file or dataset_path must be provided.")
        return self


class CodeExecutionInput(BaseModel):
    task_description: str = Field(
        ...,
        description="Clear description of the task: e.g. 'Split train.csv into train/val/test 70/15/15', 'Train a classifier on train.csv and save model', 'Load model X and predict on new_data.csv'. Required. Code is generated and executed in a sandbox.",
    )
    input_files: Optional[List[str]] = Field(
        default=None,
        description="Optional list of absolute or relative paths to input files (CSV, FASTA, etc.). These paths are passed to the generated code. Omit or empty for tasks that need no input files.",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Optional output directory for generated files. Prefer the session-scoped "
            "agent dir (e.g. temp_outputs/web_v2/sessions/{sid}/...) so artifacts stay "
            "isolated per session. When omitted, falls back to the parent dir of the "
            "first input file, or the global Code_Execution/Generated_Outputs path."
        ),
    )


class ModelTrainingInput(BaseModel):
    config_path: str = Field(
        ...,
        description="Absolute or relative path to the training configuration JSON file (e.g. from generate_training_config). File must exist. Required.",
    )


class ModelPredictionInput(BaseModel):
    config_path: str = Field(
        ...,
        description=(
            "Path to the training/prediction configuration JSON, OR a registered "
            "model_id from list_trained_models / train_protein_model registration. Required."
        ),
    )
    sequence: Optional[str] = Field(
        default=None,
        description="Single protein sequence (one-letter amino acid string) for single-sequence prediction. Required for single-sequence mode; omit when using csv_file for batch.",
    )
    csv_file: Optional[str] = Field(
        default=None,
        description="Path to CSV file containing sequences (e.g. aa_seq column) for batch prediction. Required for batch mode; omit when using sequence for single prediction. Exactly one of sequence or csv_file must be provided.",
    )

    @model_validator(mode="after")
    def require_sequence_or_csv(self):
        if not self.sequence and not self.csv_file:
            raise ValueError("Either sequence or csv_file must be provided for prediction.")
        return self


class RegisterTrainedModelInput(BaseModel):
    config_path: str = Field(
        ...,
        description="Path to the training configuration JSON used for training (or the registered copy).",
    )
    model_id: Optional[str] = Field(
        default=None,
        description="Optional stable id for ckpt/user_trained/<model_id>. Auto-generated when omitted.",
    )
    output_dir: Optional[str] = Field(
        default=None,
        description="Optional override of the training output directory containing the checkpoint.",
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Optional explicit path to the .pt/.pth checkpoint file.",
    )


def _guard(tool_name: str):
    try:
        assert_tool_allowed(tool_name)
        return None
    except RuntimeError:
        return tool_denied_json(tool_name)


@tool("generate_training_config", args_schema=CSVTrainingConfigInput)
def generate_training_config_tool(
    csv_file: Optional[str] = None,
    dataset_path: Optional[str] = None,
    valid_csv_file: Optional[str] = None,
    test_csv_file: Optional[str] = None,
    output_name: str = "custom_training_config",
    user_requirements: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> str:
    """Build training JSON from workspace CSV data and explicit parameters. Default backbone ESM2-8M; online training supports this backbone only. Returns settings and disjoint data splits for queued train_protein_model."""
    denied = _guard("generate_training_config")
    if denied:
        return denied
    try:
        return process_csv_and_generate_config(
            csv_file, valid_csv_file, test_csv_file, output_name,
            dataset_path=dataset_path, user_requirements=user_requirements, user_overrides=parameters,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": f"Training config generation error: {str(e)}"}, ensure_ascii=False)


@tool("agent_generated_code", args_schema=CodeExecutionInput)
def agent_generated_code_tool(
    task_description: str,
    input_files: Optional[List[str]] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Retired compatibility entry. Use the session-bound write and execute tools."""
    from mcp_tools.tools.tool_surface import CODE_MIGRATION_MESSAGE
    return json.dumps({'success': False, 'error': CODE_MIGRATION_MESSAGE, 'code': 'tool_retired'})


@tool("train_protein_model", args_schema=ModelTrainingInput)
def train_protein_model_tool(config_path: str) -> str:
    """Train from a workspace configuration through the compute queue. Returns fitted checkpoint and measured evaluation logs. Online supports ESM2-8M; research models remain private until independent review."""
    denied = _guard("train_protein_model")
    if denied:
        return denied
    return run_train_tool(config_path)


@tool("protein_model_predict", args_schema=ModelPredictionInput)
def protein_model_predict_tool(
    config_path: str,
    sequence: Optional[str] = None,
    csv_file: Optional[str] = None,
) -> str:
    """Predict protein properties from a trained checkpoint. Accept a single sequence or a workspace CSV. Online uses session-owned ESM2-8M models."""
    denied = _guard("protein_model_predict")
    if denied:
        return denied
    return run_predict_tool(config_path, sequence=sequence, csv_file=csv_file)


@tool("register_trained_model", args_schema=RegisterTrainedModelInput)
def register_trained_model_tool(
    config_path: str,
    model_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    model_path: Optional[str] = None,
) -> str:
    """Store an already-trained checkpoint. Research contexts stage candidates and cannot self-publish; production local registration is unchanged. Local mode only."""
    denied = _guard("register_trained_model")
    if denied:
        return denied
    try:
        return register_trained_model_json(
            config_path=config_path,
            model_id=model_id,
            output_dir=output_dir,
            model_path=model_path,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": f"Register error: {str(e)}"}, ensure_ascii=False)


@tool("list_trained_models")
def list_trained_models_tool() -> str:
    """List models registered under ckpt/user_trained (cross-session). Local mode only."""
    denied = _guard("list_trained_models")
    if denied:
        return denied
    try:
        return list_trained_models_json()
    except Exception as e:
        return json.dumps({"success": False, "error": f"List error: {str(e)}"}, ensure_ascii=False)


TRAIN_TOOLS = [
    generate_training_config_tool,
    train_protein_model_tool,
    protein_model_predict_tool,
    register_trained_model_tool,
    list_trained_models_tool,
]
