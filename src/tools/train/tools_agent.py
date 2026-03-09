# train: LangChain @tool layer; calls core (train_operations), returns JSON.

import json
from typing import List, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field, model_validator

from .train_operations import (
    process_csv_and_generate_config,
    generate_and_execute_code,
    run_train_tool,
    run_predict_tool,
)


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
        description="Optional free-text or structured requirements (e.g. 'use ProtT5 + QLoRA', '2 epochs', 'learning_rate 1e-5'). Used by AI to generate training parameters.",
    )

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


class ModelTrainingInput(BaseModel):
    config_path: str = Field(
        ...,
        description="Absolute or relative path to the training configuration JSON file (e.g. from generate_training_config). File must exist. Required.",
    )


class ModelPredictionInput(BaseModel):
    config_path: str = Field(
        ...,
        description="Path to the training/prediction configuration JSON (same config used for training). Required.",
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
def agent_generated_code_tool(task_description: str, input_files: Optional[List[str]] = None) -> str:
    """Generate and execute Python code based on task description (agent-generated code runs in a sandbox; malicious patterns are blocked). Use for data processing, splitting, analysis tasks."""
    try:
        return generate_and_execute_code(task_description, input_files)
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


TRAIN_TOOLS = [
    generate_training_config_tool,
    agent_generated_code_tool,
    train_protein_model_tool,
    protein_model_predict_tool,
]
