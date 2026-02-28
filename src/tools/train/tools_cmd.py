"""
CLI for training tools: config generation, train, predict, AI code execution.
No Gradio/MCP; invokes tools_mcp logic or runs src/train.py, src/predict.py.
Run from project root (e.g. PYTHONPATH=src python -m tools.train.tools_cmd).
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Ensure src is on path for web.utils imports
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]  # tools/train -> tools -> src
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def _import_tools_mcp():
    """Lazy import to avoid loading heavy deps before subcommand is chosen."""
    from tools.train.tools_mcp import (
        process_csv_and_generate_config,
        generate_and_execute_code,
        run_train_tool,
        run_predict_tool,
    )
    return process_csv_and_generate_config, generate_and_execute_code, run_train_tool, run_predict_tool


def cmd_generate_config(args: argparse.Namespace) -> int:
    csv_file = getattr(args, "csv_file", None)
    dataset_path = getattr(args, "dataset_path", None)
    if not csv_file and not dataset_path:
        print("Error: --csv_file or --dataset_path required", file=sys.stderr)
        return 1
    process_csv_and_generate_config, *_ = _import_tools_mcp()
    result = process_csv_and_generate_config(
        csv_file=csv_file,
        valid_csv_file=getattr(args, "valid_csv_file", None),
        test_csv_file=getattr(args, "test_csv_file", None),
        output_name=getattr(args, "output_name", "custom_training_config"),
        dataset_path=dataset_path,
        user_requirements=getattr(args, "user_requirements", None),
    )
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        return 0 if parsed.get("success") else 1
    except json.JSONDecodeError:
        print(result)
        return 0


def cmd_train(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config_path", None)
    if not config_path:
        print("Error: --config_path required", file=sys.stderr)
        return 1
    if not os.path.exists(config_path):
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        return 1
    _, _, run_train_tool, _ = _import_tools_mcp()
    result = run_train_tool(config_path)
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        return 0 if parsed.get("success") else 1
    except json.JSONDecodeError:
        print(result)
        return 0


def cmd_predict(args: argparse.Namespace) -> int:
    config_path = getattr(args, "config_path", None)
    if not config_path:
        print("Error: --config_path required", file=sys.stderr)
        return 1
    sequence = getattr(args, "sequence", None)
    csv_file = getattr(args, "csv_file", None)
    if not sequence and not csv_file:
        print("Error: --sequence or --csv_file required", file=sys.stderr)
        return 1
    _, _, _, run_predict_tool = _import_tools_mcp()
    result = run_predict_tool(config_path, sequence=sequence, csv_file=csv_file)
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        return 0 if parsed.get("success") else 1
    except json.JSONDecodeError:
        print(result)
        return 0


def cmd_agent_generated_code(args: argparse.Namespace) -> int:
    task = getattr(args, "task_description", None)
    if not task:
        print("Error: --task_description required", file=sys.stderr)
        return 1
    _, generate_and_execute_code, _, _ = _import_tools_mcp()
    input_files = getattr(args, "input_files", None)
    if input_files:
        input_files = [f.strip() for f in input_files.split(",") if f.strip()]
    result = generate_and_execute_code(task, input_files or None)
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        return 0 if parsed.get("success") else 1
    except json.JSONDecodeError:
        print(result)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Training tools CLI: config generation, train, predict, agent-generated code execution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generate-config
    p_gen = sub.add_parser("generate-config", help="Generate training config from CSV or Hugging Face dataset")
    p_gen.add_argument("--csv_file", default=None, help="Path to training CSV file")
    p_gen.add_argument("--dataset_path", default=None, help="Hugging Face dataset path (e.g. username/dataset)")
    p_gen.add_argument("--valid_csv_file", default=None, help="Optional validation CSV")
    p_gen.add_argument("--test_csv_file", default=None, help="Optional test CSV")
    p_gen.add_argument("--output_name", default="custom_training_config", help="Output config name (without .json)")
    p_gen.add_argument("--user_requirements", default=None, help="User requirements for config generation")
    p_gen.set_defaults(run=cmd_generate_config)

    # train
    p_train = sub.add_parser("train", help="Train protein model with a config file")
    p_train.add_argument("--config_path", required=True, help="Path to training config JSON")
    p_train.set_defaults(run=cmd_train)

    # predict
    p_pred = sub.add_parser("predict", help="Predict with trained model (single sequence or batch CSV)")
    p_pred.add_argument("--config_path", required=True, help="Path to training config JSON (defines model)")
    p_pred.add_argument("--sequence", default=None, help="Single protein sequence")
    p_pred.add_argument("--csv_file", default=None, help="CSV file for batch prediction")
    p_pred.set_defaults(run=cmd_predict)

    # agent-generated-code (sandboxed; malicious patterns blocked)
    p_code = sub.add_parser("agent-generated-code", help="Generate and run Python code via LLM (sandboxed, needs OPENAI_API_KEY)")
    p_code.add_argument("--task_description", required=True, help="Task description for code generation")
    p_code.add_argument("--input_files", default=None, help="Comma-separated input file paths")
    p_code.set_defaults(run=cmd_agent_generated_code)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
