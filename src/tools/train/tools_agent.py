# training: 配置生成、训练、预测、AI 代码执行的 @tool 定义，逻辑在 .tools_mcp

import json
from typing import Any, Dict, List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field, validator

from .tools_mcp import (
    process_csv_and_generate_config,
    generate_and_execute_code,
    run_train_tool,
    run_predict_tool,
)


class CSVTrainingConfigInput(BaseModel):
    csv_file: Optional[str] = Field(None, description="Path to CSV file with training data or None if using Hugging Face dataset")
    dataset_path: Optional[str] = Field(None, description="Dataset path (Local or Hugging Face path like 'username/dataset_name')")
    valid_csv_file: Optional[str] = Field(None, description="Optional path to validation CSV file")
    test_csv_file: Optional[str] = Field(None, description="Optional path to test CSV file")
    output_name: str = Field(default="custom_training_config", description="Name for the generated config")
    user_requirements: Optional[Any] = Field(None, description="User-specified training requirements (string or dict)")

    class Config:
        arbitrary_types_allowed = True

    @validator("dataset_path", "csv_file")
    def validate_input_sources(cls, v, values):
        if "csv_file" in values and values["csv_file"] is None and "dataset_path" in values and values["dataset_path"] is None:
            raise ValueError("Either csv_file or dataset_path must be provided")
        return v


class CodeExecutionInput(BaseModel):
    task_description: str = Field(..., description="Description of the task to be accomplished")
    input_files: Optional[List[str]] = Field(default=None, description="Optional list of input file paths")


class ModelTrainingInput(BaseModel):
    config_path: str = Field(..., description="Path to training configuration JSON file")


class ModelPredictionInput(BaseModel):
    config_path: str = Field(..., description="Path to prediction configuration JSON file")
    sequence: Optional[str] = Field(None, description="Single protein sequence to predict")
    csv_file: Optional[str] = Field(None, description="Path to CSV file with sequences (for batch prediction)")


@tool("generate_training_config", args_schema=CSVTrainingConfigInput)
def generate_training_config_tool(
    csv_file: Optional[str] = None,
    dataset_path: Optional[str] = None,
    valid_csv_file: Optional[str] = None,
    test_csv_file: Optional[str] = None,
    output_name: str = "custom_training_config",
    user_requirements: Optional[str] = None,
) -> str:
    """Generate training JSON configuration from CSV files or Hugging Face datasets containing protein sequences and labels."""
    try:
        return process_csv_and_generate_config(
            csv_file, valid_csv_file, test_csv_file, output_name,
            dataset_path=dataset_path, user_requirements=user_requirements,
        )
    except Exception as e:
        return json.dumps({"success": False, "error": f"Training config generation error: {str(e)}"}, ensure_ascii=False)


@tool("agent_generated_code", args_schema=CodeExecutionInput)
def agent_generated_code_tool(task_description: str, input_files: List[str] = []) -> str:
    """Generate and execute Python code based on task description (agent-generated code runs in a sandbox; malicious patterns are blocked). Use for data processing, splitting, analysis tasks."""
    try:
        return generate_and_execute_code(task_description, input_files or None)
    except Exception as e:
        return json.dumps({"success": False, "error": f"Code execution error: {str(e)}"}, ensure_ascii=False)


@tool("train_protein_model", args_schema=ModelTrainingInput)
def train_protein_model_tool(config_path: str) -> str:
    """Train a protein language model using a configuration file. Executes the training process and streams logs."""
    return run_train_tool(config_path)


@tool("protein_model_predict", args_schema=ModelPredictionInput)
def protein_model_predict_tool(
    config_path: str,
    sequence: Optional[str] = None,
    csv_file: Optional[str] = None,
) -> str:
    """Predict protein properties using a trained model. Single sequence or batch from CSV."""
    return run_predict_tool(config_path, sequence=sequence, csv_file=csv_file)
