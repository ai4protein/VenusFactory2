"""
CLI for file tools: MAXIT, FASTA/UID, archive, PDB, metadata.
Run from project root: PYTHONPATH=src python -m tools.file.tools_cmd ...
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[2]
_SRC_DIR = _REPO_ROOT / "src"


def _env_with_src():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_SRC_DIR)
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"] + os.pathsep + env["PYTHONPATH"]
    return env


def _run_maxit(args_list: list[str]) -> int:
    module = "tools.file.converter.maxit"
    cmd = [sys.executable, "-m", module] + args_list
    return subprocess.run(cmd, cwd=str(_REPO_ROOT), env=_env_with_src()).returncode


def cmd_maxit(args: argparse.Namespace) -> int:
    strategy = getattr(args, "strategy", None)
    if not strategy:
        print("Error: --strategy required (pdb2cif, cif2pdb, cif2mmcif)", file=sys.stderr)
        return 1
    cmd_args = ["--strategy", strategy]
    if getattr(args, "file", None):
        cmd_args += ["--file", args.file]
    if getattr(args, "input_dir", None):
        cmd_args += ["--input_dir", args.input_dir]
    if getattr(args, "out_dir", None):
        cmd_args += ["--out_dir", args.out_dir]
    if not getattr(args, "file", None) and not getattr(args, "input_dir", None):
        print("Error: --file or --input_dir required", file=sys.stderr)
        return 1
    return _run_maxit(cmd_args)


def _call_file_tool(call_fn, **kwargs) -> int:
    """Invoke a call_* from tools_mcp and print JSON result; return 0 on success."""
    try:
        from tools.file.tools_mcp import call_read_fasta, call_extract_uids_from_fasta, call_uid_file_to_chunks
        from tools.file.tools_mcp import call_unzip, call_ungzip
        from tools.file.tools_mcp import call_pdb_chain_sequences, call_pdb_dir_to_fasta, call_pdb_is_apo
        from tools.file.tools_mcp import call_rcsb_metadata_uniprot_id
        fns = {
            "read_fasta": call_read_fasta,
            "extract_uids": call_extract_uids_from_fasta,
            "uid_chunks": call_uid_file_to_chunks,
            "unzip": call_unzip,
            "ungzip": call_ungzip,
            "pdb_chains": call_pdb_chain_sequences,
            "pdb_to_fasta": call_pdb_dir_to_fasta,
            "pdb_is_apo": call_pdb_is_apo,
            "rcsb_uniprot": call_rcsb_metadata_uniprot_id,
        }
        fn = fns.get(call_fn)
        if not fn:
            return 1
        out = fn(**kwargs)
        print(out)
        import json
        d = json.loads(out)
        return 0 if d.get("success") else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_read_fasta(args: argparse.Namespace) -> int:
    return _call_file_tool("read_fasta", file_path=args.file_path)


def cmd_extract_uids(args: argparse.Namespace) -> int:
    return _call_file_tool(
        "extract_uids",
        multi_fasta_file=args.multi_fasta_file,
        uid_file=getattr(args, "uid_file", None),
        separator=getattr(args, "separator", "|"),
        uid_index=getattr(args, "uid_index", 1),
    )


def cmd_uid_chunks(args: argparse.Namespace) -> int:
    return _call_file_tool(
        "uid_chunks",
        uid_file=args.uid_file,
        chunk_dir=getattr(args, "chunk_dir", None),
        chunk_size=getattr(args, "chunk_size", 10000),
    )


def cmd_unzip(args: argparse.Namespace) -> int:
    return _call_file_tool("unzip", zip_path=args.zip_path, save_folder=args.save_folder)


def cmd_ungzip(args: argparse.Namespace) -> int:
    return _call_file_tool("ungzip", gz_path=args.gz_path, out_dir=args.out_dir)


def cmd_pdb_chains(args: argparse.Namespace) -> int:
    return _call_file_tool("pdb_chains", pdb_file=args.pdb_file)


def cmd_pdb_to_fasta(args: argparse.Namespace) -> int:
    return _call_file_tool(
        "pdb_to_fasta",
        pdb_dir=args.pdb_dir,
        out_fasta_path=args.out_fasta_path,
        use_chain_a_only=not getattr(args, "no_chain_a_only", False),
    )


def cmd_pdb_is_apo(args: argparse.Namespace) -> int:
    return _call_file_tool("pdb_is_apo", pdb_path=args.pdb_path)


def cmd_rcsb_uniprot(args: argparse.Namespace) -> int:
    return _call_file_tool("rcsb_uniprot", meta_data_file=args.meta_data_file)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="File tools CLI: MAXIT, FASTA/UID, archive, PDB, metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("maxit", help="Convert structure files with MAXIT (pdb2cif, cif2pdb, cif2mmcif)")
    p.add_argument("--strategy", required=True, choices=["pdb2cif", "cif2pdb", "cif2mmcif"], help="Conversion type")
    p.add_argument("--file", help="Single input file path")
    p.add_argument("--input_dir", help="Input directory (convert all .pdb/.cif inside)")
    p.add_argument("--out_dir", help="Output directory")
    p.set_defaults(run=cmd_maxit)

    p = sub.add_parser("read-fasta", help="Parse multi-FASTA and output JSON")
    p.add_argument("--file_path", required=True, help="Path to FASTA file")
    p.set_defaults(run=cmd_read_fasta)

    p = sub.add_parser("extract-uids", help="Extract IDs from FASTA headers (e.g. UniProt)")
    p.add_argument("--multi_fasta_file", required=True, help="Multi-FASTA file")
    p.add_argument("--uid_file", help="Optional output file for UID list")
    p.add_argument("--separator", default="|", help="Header field separator")
    p.add_argument("--uid_index", type=int, default=1, help="0-based index of ID in header")
    p.set_defaults(run=cmd_extract_uids)

    p = sub.add_parser("uid-chunks", help="Split UID list file into chunk files")
    p.add_argument("--uid_file", required=True, help="File with one ID per line")
    p.add_argument("--chunk_dir", help="Output directory for chunks (default: <uid_file_dir>/chunks)")
    p.add_argument("--chunk_size", type=int, default=10000, help="IDs per chunk")
    p.set_defaults(run=cmd_uid_chunks)

    p = sub.add_parser("unzip", help="Extract zip archive")
    p.add_argument("--zip_path", required=True, help="Path to .zip file")
    p.add_argument("--save_folder", required=True, help="Extract destination")
    p.set_defaults(run=cmd_unzip)

    p = sub.add_parser("ungzip", help="Decompress .gz file")
    p.add_argument("--gz_path", required=True, help="Path to .gz file")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.set_defaults(run=cmd_ungzip)

    p = sub.add_parser("pdb-chains", help="Extract chain sequences from PDB")
    p.add_argument("--pdb_file", required=True, help="Path to PDB file")
    p.set_defaults(run=cmd_pdb_chains)

    p = sub.add_parser("pdb-to-fasta", help="Write FASTA from all PDBs in directory")
    p.add_argument("--pdb_dir", required=True, help="Directory of PDB files")
    p.add_argument("--out_fasta_path", required=True, help="Output FASTA path")
    p.add_argument("--no_chain_a_only", action="store_true", help="Use first chain instead of chain A")
    p.set_defaults(run=cmd_pdb_to_fasta)

    p = sub.add_parser("pdb-is-apo", help="Check if PDB is apo (no hetero residues)")
    p.add_argument("--pdb_path", required=True, help="Path to PDB file")
    p.set_defaults(run=cmd_pdb_is_apo)

    p = sub.add_parser("rcsb-uniprot", help="Get UniProt ID from RCSB metadata JSON")
    p.add_argument("--meta_data_file", required=True, help="Path to RCSB metadata JSON")
    p.set_defaults(run=cmd_rcsb_uniprot)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
