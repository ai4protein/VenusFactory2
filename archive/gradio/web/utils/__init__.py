"""Web utility exports loaded only when their owning feature is requested.

Scientific tools can import path helpers without loading the UI stack. Existing
package exports retain the legacy module order, including the public fallback.
"""

from importlib import import_module as _import_module

_MODULES = (
    "constants",
    "common_utils",
    "file_handlers",
    "llm_helpers",
    "data_processors",
    "visualization",
    "prediction_runners",
    "chat_helpers",
    "label_mappers",
    "ui_helpers",
)

_EXPORTS = {
    "MODEL_MAPPING_ZERO_SHOT": "prediction_runners",
    "DATASET_MAPPING_ZERO_SHOT": "constants",
    "MODEL_MAPPING_FUNCTION": "constants",
    "MODEL_ADAPTER_MAPPING_FUNCTION": "constants",
    "MODEL_RESIDUE_MAPPING_FUNCTION": "constants",
    "DATASET_MAPPING_FUNCTION": "data_processors",
    "LABEL_MAPPING_FUNCTION": "label_mappers",
    "COLOR_MAP_FUNCTION": "visualization",
    "PROTEIN_PROPERTIES_FUNCTION": "constants",
    "PROTEIN_PROPERTIES_MAP_FUNCTION": "prediction_runners",
    "RESIDUE_MAPPING_FUNCTION": "constants",
    "REGRESSION_TASKS_FUNCTION": "label_mappers",
    "REGRESSION_TASKS_FUNCTION_MAX_MIN": "label_mappers",
    "LLM_MODELS": "llm_helpers",
    "DATASET_TO_TASK_MAP": "label_mappers",
    "sanitize_filename": "common_utils",
    "get_save_path": "common_utils",
    "toggle_ai_section": "common_utils",
    "format_physical_chemical_properties": "common_utils",
    "format_rsa_results": "common_utils",
    "format_sasa_results": "common_utils",
    "format_secondary_structure_results": "common_utils",
    "extract_sequence_from_pdb": "file_handlers",
    "parse_fasta_paste_content": "ui_helpers",
    "save_selected_sequence_fasta": "file_handlers",
    "parse_pdb_paste_content": "ui_helpers",
    "save_selected_chain_pdb": "file_handlers",
    "handle_paste_chain_selection": "file_handlers",
    "handle_paste_sequence_selection": "file_handlers",
    "handle_pdb_chain_change": "file_handlers",
    "handle_fasta_sequence_change": "file_handlers",
    "process_pdb_file_upload": "file_handlers",
    "process_fasta_file_upload": "file_handlers",
    "handle_file_upload": "file_handlers",
    "process_fasta_file": "file_handlers",
    "clear_paste_content_pdb": "file_handlers",
    "clear_paste_content_fasta": "file_handlers",
    "handle_sequence_change_unified": "file_handlers",
    "get_chat_base_url": "llm_helpers",
    "get_api_key": "llm_helpers",
    "LLMConfig": "llm_helpers",
    "call_llm_api": "llm_helpers",
    "check_llm_config_status": "llm_helpers",
    "on_llm_change": "llm_helpers",
    "generate_expert_analysis_prompt_residue": "llm_helpers",
    "generate_expert_analysis_prompt": "llm_helpers",
    "format_expert_response": "llm_helpers",
    "generate_mutation_ai_prompt": "llm_helpers",
    "generate_ai_summary_prompt": "llm_helpers",
    "get_total_residues_count": "data_processors",
    "update_dataset_choices": "data_processors",
    "expand_residue_predictions": "data_processors",
    "prepare_top_residue_heatmap_data": "data_processors",
    "perform_soft_voting": "data_processors",
    "map_prediction_labels": "data_processors",
    "generate_plotly_heatmap": "visualization",
    "generate_plots_for_all_results": "visualization",
    "generate_plots_for_residue_results": "visualization",
    "run_zero_shot_prediction": "prediction_runners",
    "run_single_function_prediction": "prediction_runners",
    "run_protein_properties_prediction": "prediction_runners",
    "make_chat_html": "chat_helpers",
    "export_chat_history_html": "chat_helpers",
    "save_chat_history_to_server": "chat_helpers",
    "map_labels": "label_mappers",
    "map_labels_individual": "label_mappers",
    "create_progress_html": "ui_helpers",
    "create_status_html": "ui_helpers",
    "handle_paste_fasta_detect": "ui_helpers",
    "handle_paste_pdb_detect": "ui_helpers",
    "update_dataset_choices_fixed": "ui_helpers",
    "wrap_upload_show_section": "ui_helpers",
    "wrap_paste_show_section": "ui_helpers",
    "wrap_clear_hide_section": "ui_helpers",
}

__all__ = [
    "MODEL_MAPPING_ZERO_SHOT",
    "DATASET_MAPPING_ZERO_SHOT",
    "MODEL_MAPPING_FUNCTION",
    "MODEL_ADAPTER_MAPPING_FUNCTION",
    "MODEL_RESIDUE_MAPPING_FUNCTION",
    "DATASET_MAPPING_FUNCTION",
    "LABEL_MAPPING_FUNCTION",
    "COLOR_MAP_FUNCTION",
    "PROTEIN_PROPERTIES_FUNCTION",
    "PROTEIN_PROPERTIES_MAP_FUNCTION",
    "RESIDUE_MAPPING_FUNCTION",
    "REGRESSION_TASKS_FUNCTION",
    "REGRESSION_TASKS_FUNCTION_MAX_MIN",
    "DATASET_TO_TASK_MAP",
    "LLM_MODELS",
    "sanitize_filename",
    "get_save_path",
    "toggle_ai_section",
    "create_zip_archive",
    "format_physical_chemical_properties",
    "format_rsa_results",
    "format_sasa_results",
    "format_secondary_structure_results",
    "parse_fasta_paste_content",
    "save_selected_sequence_fasta",
    "parse_pdb_paste_content",
    "save_selected_chain_pdb",
    "handle_paste_chain_selection",
    "handle_paste_sequence_selection",
    "handle_pdb_chain_change",
    "handle_fasta_sequence_change",
    "process_pdb_file_upload",
    "process_fasta_file_upload",
    "handle_file_upload",
    "process_fasta_file",
    "clear_paste_content_pdb",
    "clear_paste_content_fasta",
    "handle_sequence_change_unified",
    "LLMConfig",
    "get_api_key",
    "get_chat_base_url",
    "call_llm_api",
    "check_llm_config_status",
    "on_llm_change",
    "generate_expert_analysis_prompt_residue",
    "generate_expert_analysis_prompt",
    "format_expert_response",
    "generate_mutation_ai_prompt",
    "generate_ai_summary_prompt",
    "get_total_residues_count",
    "update_dataset_choices",
    "expand_residue_predictions",
    "prepare_top_residue_heatmap_data",
    "map_prediction_labels",
    "perform_soft_voting",
    "generate_plotly_heatmap",
    "generate_plots_for_all_results",
    "generate_plots_for_residue_results",
    "run_zero_shot_prediction",
    "run_single_function_prediction",
    "run_protein_properties_prediction",
    "send_feedback_email",
    "handle_feedback_submit",
    "make_chat_html",
    "export_chat_history_html",
    "save_chat_history_to_server",
    "map_labels",
    "map_labels_individual",
    "create_progress_html",
    "create_status_html",
    "handle_paste_fasta_detect",
    "handle_paste_pdb_detect",
    "update_dataset_choices_fixed",
    "wrap_upload_show_section",
    "wrap_paste_show_section",
    "wrap_clear_hide_section",
    "extract_sequence_from_pdb",
]


def __getattr__(name):
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _MODULES:
        value = _import_module(f"{__name__}.{name}")
    elif name in _EXPORTS:
        value = getattr(_import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    else:
        # Preserve unlisted legacy public exports without eagerly importing all
        # UI modules for ordinary direct submodule imports.
        for module_name in reversed(_MODULES):
            module = _import_module(f"{__name__}.{module_name}")
            exports = getattr(module, "__all__", None)
            if (exports is None or name in exports) and hasattr(module, name):
                value = getattr(module, name)
                break
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | set(_MODULES))
