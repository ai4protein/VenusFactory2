"""
tools/denovo/tools_cmd.py — ProteinMPNN 命令行接口

两个子命令：design / score
"""
import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MPNN_DIR = _HERE / "proteinmpnn"
if str(_MPNN_DIR) not in sys.path:
    sys.path.insert(0, str(_MPNN_DIR))

from protein_mpnn_function import proteinmpnn_design, proteinmpnn_score


def cmd_design(args: argparse.Namespace) -> int:
    fixed_residues = None
    if args.fixed_residues_json:
        fixed_residues = json.loads(args.fixed_residues_json)
    fasta = proteinmpnn_design(
        pdb_path=args.pdb_path,
        designed_chains=args.designed_chains,
        fixed_chains=args.fixed_chains,
        fixed_residues=fixed_residues,
        homomer=args.homomer,
        num_sequences=args.num_sequences,
        temperatures=args.temperatures,
        omit_aas=args.omit_aas,
        model_name=args.model_name,
        backbone_noise=args.backbone_noise,
        ca_only=args.ca_only,
    )
    print(f"Done. FASTA: {fasta}")
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    score_dir = proteinmpnn_score(
        pdb_path=args.pdb_path,
        fasta_path=args.fasta_path,
        designed_chains=args.designed_chains,
        num_batches=args.num_batches,
        model_name=args.model_name,
        backbone_noise=args.backbone_noise,
    )
    print(f"Done. Scores: {score_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ProteinMPNN CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pd = sub.add_parser("design", help="Sequence design (single/multi-chain/interface/homomer/partial)")
    pd.add_argument("--pdb_path", required=True)
    pd.add_argument("--designed_chains", nargs="+", default=None, help="Chains to redesign; None = all")
    pd.add_argument("--fixed_chains", nargs="+", default=None, help="Fixed context chains (interface design)")
    pd.add_argument("--fixed_residues_json", default=None, help='e.g. \'{"A": [1, 5, 10]}\'')
    pd.add_argument("--homomer", action="store_true", help="Enforce symmetric design across chains")
    pd.add_argument("--num_sequences", type=int, default=8)
    pd.add_argument("--temperatures", nargs="+", type=float, default=None, help="e.g. 0.1 0.2")
    pd.add_argument("--omit_aas", default="X")
    pd.add_argument("--model_name", default="v_48_020")
    pd.add_argument("--backbone_noise", type=float, default=0.0)
    pd.add_argument("--ca_only", action="store_true")
    pd.set_defaults(run=cmd_design)

    ps = sub.add_parser("score", help="Score sequences against backbone (NLL)")
    ps.add_argument("--pdb_path", required=True)
    ps.add_argument("--fasta_path", default=None, help="FASTA to score; None = native sequence")
    ps.add_argument("--designed_chains", nargs="+", default=None)
    ps.add_argument("--num_batches", type=int, default=1)
    ps.add_argument("--model_name", default="v_48_020")
    ps.add_argument("--backbone_noise", type=float, default=0.0)
    ps.set_defaults(run=cmd_score)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
