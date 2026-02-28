"""
CLI for predict tools: standard command-line invocation of local physchem/finetuned scripts.
No Gradio/MCP; runs src/tools/predict/features/*.py and finetuned/*.py.
Config: src/constant.json["filter_tools"].
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Repo root when this file is .../src/tools/predict/tools_cmd.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]

# Load filter_tools config from constant.json
_CONSTANT_PATH = _REPO_ROOT / "src" / "constant.json"
with open(_CONSTANT_PATH, "r", encoding="utf-8") as f:
    _FILTER_CFG = json.load(f)["filter_tools"]

# Subcommand -> (script path relative to repo root, PDB-only?)
FEATURES_SCRIPTS = {k: (v[0], v[1]) for k, v in _FILTER_CFG["features_scripts"].items()}
FINETUNED_SCRIPTS = _FILTER_CFG["finetuned_scripts"]
FINETUNED_TASKS = _FILTER_CFG["finetuned_tasks"]
CKPT_BASE = _FILTER_CFG["ckpt_base"]
MODEL_NAME_TO_CKPT_DIR = _FILTER_CFG["model_name_to_ckpt_dir"]


def _run_script(script_rel: str, args: list[str]) -> int:
    script_path = _REPO_ROOT / script_rel
    if not script_path.exists():
        print(f"Error: script not found {script_path}", file=sys.stderr)
        return 1
    env = os.environ.copy()
    repo_root = str(_REPO_ROOT)
    env["PYTHONPATH"] = repo_root + (os.pathsep + env.get("PYTHONPATH", ""))
    cmd = [sys.executable, str(script_path)] + args
    return subprocess.run(cmd, cwd=repo_root, env=env).returncode


def cmd_protein_properties(args: argparse.Namespace) -> int:
    task = getattr(args, "task", "physchem").lower().replace(" ", "-")
    if task == "physical-and-chemical-properties":
        task = "physchem"
    if task not in FEATURES_SCRIPTS:
        print(f"Error: unknown task {args.task}. Choose from: {list(FEATURES_SCRIPTS)}", file=sys.stderr)
        return 1
    script_rel, pdb_only = FEATURES_SCRIPTS[task]
    fasta = getattr(args, "fasta_file", None)
    pdb = getattr(args, "pdb_file", None)
    inp = getattr(args, "input_file", None)
    chain = getattr(args, "chain_id", "A") or "A"
    out = getattr(args, "output_file", None)

    if task == "all":
        if not inp and not fasta and not pdb:
            print("Error: --input_file (or --fasta_file / --pdb_file) required", file=sys.stderr)
            return 1
        cmd_args = ["--input_file", inp or (pdb or fasta)]
        if pdb or (inp and inp.lower().endswith(".pdb")):
            cmd_args += ["--file_type", "pdb", "--chain_id", chain]
        if out:
            cmd_args += ["--output_file", out]
        if getattr(args, "print_summary", False):
            cmd_args += ["--print_summary"]
        return _run_script(script_rel, cmd_args)

    cmd_args = []
    if pdb_only:
        if not pdb:
            print("Error: --pdb_file required for this task", file=sys.stderr)
            return 1
        cmd_args = ["--pdb_file", pdb, "--chain_id", chain]
    else:
        if fasta:
            cmd_args = ["--fasta_file", fasta]
        elif pdb:
            cmd_args = ["--pdb_file", pdb, "--chain_id", chain]
        else:
            print("Error: --fasta_file or --pdb_file required", file=sys.stderr)
            return 1
    if out:
        cmd_args += ["--output_file", out]
    return _run_script(script_rel, cmd_args)


def _model_name_to_backbone(model_name: str) -> str:
    """Map checkpoint model name to script key (ankh, prott5, protbert, esm2)."""
    m = model_name.lower()
    if m.startswith("ankh"):
        return "ankh"
    if m.startswith("prott5") or "prot_t5" in m or "prot-t5" in m:
        return "prott5"
    if m.startswith("protbert") or "prot_bert" in m:
        return "protbert"
    if m.startswith("esm2"):
        return "esm2"
    return "ankh"


def cmd_finetuned(args: argparse.Namespace) -> int:
    task = getattr(args, "task", None)
    model_name = getattr(args, "model_name", "ankh-large")
    if not task:
        print("Error: --task required (dataset name, e.g. DeepET_Topt, DeepSol)", file=sys.stderr)
        return 1
    if task not in FINETUNED_TASKS:
        print(f"Warning: --task {task} not in predefined list; using ckpt/{task}/{model_name}/ anyway.", file=sys.stderr)
    fasta = getattr(args, "fasta_file", None)
    if not fasta:
        print("Error: --fasta_file required", file=sys.stderr)
        return 1
    # Adapter dir: ckpt/{task}/{ckpt_dir}/ (e.g. ckpt/DeepET_Topt/esm2_t33_650M_UR50D/)
    ckpt_dir = MODEL_NAME_TO_CKPT_DIR.get(model_name, model_name)
    adapter_path = Path(_REPO_ROOT) / CKPT_BASE / task / ckpt_dir
    if not adapter_path.exists():
        print(f"Error: checkpoint dir not found: {adapter_path}", file=sys.stderr)
        return 1
    backbone = _model_name_to_backbone(model_name)
    script_rel = FINETUNED_SCRIPTS[backbone]
    out = getattr(args, "output_csv", "prediction_results.csv")
    cmd_args = ["--fasta_file", fasta, "--adapter_path", str(adapter_path), "--output_csv", out]
    return _run_script(script_rel, cmd_args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Filter tools CLI: run local physchem/finetuned scripts (no Gradio).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # features: physchem, rsa, sasa, secondary-structure, all
    p_prop = sub.add_parser(
        "features",
        help="Protein property prediction: physchem, rsa, sasa, secondary-structure, or all",
    )
    p_prop.add_argument(
        "--task",
        default="physchem",
        choices=list(FEATURES_SCRIPTS),
        help="Task: physchem (FASTA/PDB), rsa/sasa/secondary-structure (PDB only), all (FASTA/PDB)",
    )
    p_prop.add_argument("--fasta_file", default=None, help="Path to FASTA file")
    p_prop.add_argument("--pdb_file", default=None, help="Path to PDB file (required for rsa, sasa, secondary-structure)")
    p_prop.add_argument("--input_file", default=None, help="Input file for task=all (FASTA or PDB)")
    p_prop.add_argument("--chain_id", default="A", help="Chain ID for PDB")
    p_prop.add_argument("--output_file", default=None, help="Output JSON path")
    p_prop.add_argument("--print_summary", action="store_true", help="For task=all: print human-readable summary")
    p_prop.set_defaults(run=cmd_protein_properties)

    # finetuned: specify task (dataset) and model_name; adapter path = ckpt/{task}/{model_name}/
    p_ft = sub.add_parser("finetuned", help="Finetuned prediction: --task (dataset) and --model_name (e.g. ankh-large); uses ckpt/<task>/<model_name>/")
    p_ft.add_argument("--task", required=True, help=f"Dataset/task name, e.g. {', '.join(FINETUNED_TASKS[:5])}...")
    p_ft.add_argument("--model_name", default="ankh-large", help="Backbone variant under ckpt/task/, e.g. ankh-large, ProtT5-xl-uniref50")
    p_ft.add_argument("--fasta_file", required=True, help="Path to FASTA file")
    p_ft.add_argument("--output_csv", default="prediction_results.csv", help="Output CSV path")
    p_ft.set_defaults(run=cmd_finetuned)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
