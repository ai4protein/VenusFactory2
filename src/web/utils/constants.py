"""Constants and configuration mappings for VenusFactory. Loaded from src/constant.json."""

import json
from pathlib import Path

_CONSTANT_PATH = Path(__file__).resolve().parent.parent.parent / "constant.json"
_WEB_UI = {}

if _CONSTANT_PATH.exists():
    with open(_CONSTANT_PATH, "r", encoding="utf-8") as f:
        _data = json.load(f)
    _WEB_UI = _data.get("web_ui", {})

MODEL_MAPPING_ZERO_SHOT = _WEB_UI.get("model_mapping_zero_shot", {})
DATASET_MAPPING_ZERO_SHOT = _WEB_UI.get("dataset_mapping_zero_shot", [])
MODEL_MAPPING_FUNCTION = _WEB_UI.get("model_mapping_function", {})
MODEL_ADAPTER_MAPPING_FUNCTION = _WEB_UI.get("model_adapter_mapping_function", {})
MODEL_RESIDUE_MAPPING_FUNCTION = _WEB_UI.get("model_residue_mapping_function", {})
DATASET_MAPPING_FUNCTION = _WEB_UI.get("dataset_mapping_function", {})
LABEL_MAPPING_FUNCTION = _WEB_UI.get("label_mapping_function", {})
COLOR_MAP_FUNCTION = _WEB_UI.get("color_map_function", {})
PROTEIN_PROPERTIES_FUNCTION = _WEB_UI.get("protein_properties_function", frozenset())
PROTEIN_PROPERTIES_MAP_FUNCTION = _WEB_UI.get("protein_properties_map_function", {})
RESIDUE_MAPPING_FUNCTION = _WEB_UI.get("residue_mapping_function", {})
REGRESSION_TASKS_FUNCTION = _WEB_UI.get("regression_tasks_function", [])
REGRESSION_TASKS_FUNCTION_MAX_MIN = _WEB_UI.get("regression_tasks_function_max_min", {})
LLM_MODELS = _WEB_UI.get("llm_models", {})

# Derived from DATASET_MAPPING_FUNCTION
DATASET_TO_TASK_MAP = {
    dataset: task
    for task, datasets in DATASET_MAPPING_FUNCTION.items()
    for dataset in datasets
}
