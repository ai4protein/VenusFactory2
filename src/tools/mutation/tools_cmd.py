"""
CLI for mutation tools: standard command-line invocation of local model scripts.
No Gradio/MCP; runs src/tools/mutation/models/*.py with argparse.
Config: src/constant.json["evolution_tools"].
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Repo root: .../VenusFactory when this file is .../src/tools/mutation/tools_cmd.py
_SCRIPT_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _SCRIPT_DIR / "models"
_REPO_ROOT = _SCRIPT_DIR.parents[2]  # mutation->tools->src->VenusFactory

# Model name -> script filename (from constant.json)
with open(_REPO_ROOT / "src" / "constant.json", "r", encoding="utf-8") as f:
    _EVO_CFG = json.load(f)["evolution_tools"]
SEQUENCE_MODELS = _EVO_CFG["sequence_models"]
STRUCTURE_MODELS = _EVO_CFG["structure_models"]


def _run_script(script_name: str, args: list[str]) -> int:
    script_path = _MODELS_DIR / script_name
    if not script_path.exists():
        print(f"Error: script not found {script_path}", file=sys.stderr)
        return 1
    env = os.environ.copy()
    repo_root = str(_REPO_ROOT)
    env["PYTHONPATH"] = repo_root + (os.pathsep + env.get("PYTHONPATH", ""))
    cmd = [sys.executable, str(script_path)] + args
    return subprocess.run(cmd, cwd=repo_root, env=env).returncode


def cmd_zero_shot_sequence(args: argparse.Namespace) -> int:
    fasta = getattr(args, "fasta_file", None)
    if not fasta:
        print("Error: --fasta_file required", file=sys.stderr)
        return 1
    model = getattr(args, "model_name", "ESM2-650M")
    script = SEQUENCE_MODELS.get(model)
    if not script:
        print(f"Error: unknown model {model}. Choose from: {list(SEQUENCE_MODELS)}", file=sys.stderr)
        return 1
    cmd_args = ["--fasta_file", fasta]
    if getattr(args, "mutations_csv", None):
        cmd_args += ["--mutations_csv", args.mutations_csv]
    if getattr(args, "output_csv", None):
        cmd_args += ["--output_csv", args.output_csv]
    return _run_script(script, cmd_args)


def cmd_zero_shot_structure(args: argparse.Namespace) -> int:
    pdb = getattr(args, "structure_file", None) or getattr(args, "pdb_file", None)
    if not pdb:
        print("Error: --structure_file or --pdb_file required", file=sys.stderr)
        return 1
    model = getattr(args, "model_name", "ESM-IF1")
    script = STRUCTURE_MODELS.get(model)
    if not script:
        print(f"Error: unknown model {model}. Choose from: {list(STRUCTURE_MODELS)}", file=sys.stderr)
        return 1
    cmd_args = ["--pdb_file", pdb]
    if getattr(args, "mutations_csv", None):
        cmd_args += ["--mutations_csv", args.mutations_csv]
    if getattr(args, "output_csv", None):
        cmd_args += ["--output_csv", args.output_csv]
    if script == "mifst.py" and getattr(args, "model_location", None):
        cmd_args += ["--model_location", args.model_location]
    if script == "saprot.py" and getattr(args, "chain", None):
        cmd_args += ["--chain", args.chain]
    if script == "esmif1.py" and getattr(args, "exhaustive", False):
        cmd_args += ["--exhaustive"]
    return _run_script(script, cmd_args)


def cmd_easy_mutation(args: argparse.Namespace) -> int:
    pdb = getattr(args, "pdb_file", None) or getattr(args, "structure_file", None)
    if not pdb:
        print("Error: --pdb_file or --structure_file required", file=sys.stderr)
        return 1
    cmd_args = ["--pdb_file", pdb]
    if getattr(args, "output_dir", None):
        cmd_args += ["--output_dir", args.output_dir]
    if getattr(args, "num_recommendations", None) is not None:
        cmd_args += ["--num_recommendations", str(args.num_recommendations)]
    if getattr(args, "mutations_csv", None):
        cmd_args += ["--mutations_csv", args.mutations_csv]
    return _run_script("easy_mutation.py", cmd_args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evolution tools CLI: run local model scripts (no Gradio).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # zero-shot-sequence
    p_seq = sub.add_parser("zero-shot-sequence", help="Sequence-based mutation scoring (ESM-1b, ESM2, ESM-1v, VenusPLM)")
    p_seq.add_argument("--fasta_file", required=True, help="Path to FASTA file")
    p_seq.add_argument("--model_name", default="ESM2-650M", choices=list(SEQUENCE_MODELS), help="Model to run")
    p_seq.add_argument("--mutations_csv", default=None, help="Optional mutations CSV")
    p_seq.add_argument("--output_csv", default=None, help="Output CSV path")
    p_seq.set_defaults(run=cmd_zero_shot_sequence)

    # zero-shot-structure
    p_struct = sub.add_parser("zero-shot-structure", help="Structure-based mutation scoring (ESM-IF1, MIF-ST, ProtSSN, etc.)")
    p_struct.add_argument("--structure_file", default=None, help="Path to PDB file (alias: --pdb_file)")
    p_struct.add_argument("--pdb_file", default=None, help="Path to PDB file")
    p_struct.add_argument("--model_name", default="ESM-IF1", choices=list(STRUCTURE_MODELS), help="Model to run")
    p_struct.add_argument("--mutations_csv", default=None, help="Optional mutations CSV")
    p_struct.add_argument("--output_csv", default=None, help="Output CSV path")
    p_struct.add_argument("--model_location", default=None, help="MIF-ST only: model path/name")
    p_struct.add_argument("--chain", default=None, help="SaProt only: chain id")
    p_struct.add_argument("--exhaustive", action="store_true", help="ESM-IF1 only: exhaustive mode")
    p_struct.set_defaults(run=cmd_zero_shot_structure)

    # easy-mutation (ensemble)
    p_easy = sub.add_parser("easy-mutation", help="Ensemble structure-based prediction (easy_mutation.py)")
    p_easy.add_argument("--pdb_file", default=None, help="Path to PDB file")
    p_easy.add_argument("--structure_file", default=None, help="Path to PDB file (alias)")
    p_easy.add_argument("--output_dir", default=None, help="Output directory")
    p_easy.add_argument("--num_recommendations", type=int, default=30, help="Number of recommendations")
    p_easy.add_argument("--mutations_csv", default=None, help="Optional mutations CSV")
    p_easy.set_defaults(run=cmd_easy_mutation)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
