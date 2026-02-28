"""
CLI for search tools: standard command-line invocation of local source/database scripts.
No Gradio; runs scripts via _run_script.
Run from project root: PYTHONPATH=src python -m tools.search.tools_cmd <command> [args]

Usage:
  literature   --query, --max_results, --source (arxiv,pubmed,...)
  dataset      --query, --max_results, --source (github,hugging_face)
  web-search   --query, --max_results, --source (duckduckgo,tavily)
  foldseek     --pdb_file, --output_dir, --protect_start, --protect_end
  uniprot      -i/--uniprot_id or -f/--file, -o/--out_dir, [-m/--merge], -e/--error_file
  ncbi         -i/--id or -f/--file, -o/--out_dir, [-m/--merge], -e/--error_file
  alphafold    -i/--uniprot_id or -f/--uniprot_id_file, -o/--out_dir, -e/--error_file
  rcsb         -i/--pdb_id or -f/--pdb_id_file, -o/--out_dir, -t/--type, -u/--unzip, -e/--error_file
  interpro     -i/--interpro_id or -f/--interpro_id_file, -o/--out_dir, -e/--error_file
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
    # So that scripts find the tools package
    env["PYTHONPATH"] = str(_SRC_DIR)
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"] + os.pathsep + env["PYTHONPATH"]
    return env


def _run_script(script_rel: str, args: list[str]) -> int:
    script_path = _REPO_ROOT / script_rel
    if not script_path.exists():
        script_path = _SRC_DIR / script_rel
    if not script_path.exists():
        print(f"Error: script not found {script_rel}", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(script_path)] + args
    return subprocess.run(cmd, cwd=str(_REPO_ROOT), env=_env_with_src()).returncode


def cmd_literature(args: argparse.Namespace) -> int:
    q = getattr(args, "query", None) or "protein structure prediction"
    n = getattr(args, "max_results", 5)
    src = getattr(args, "source", "arxiv,pubmed")
    return _run_script("src/tools/search/source/literature.py", ["--query", q, "--max_results", str(n), "--source", src])


def cmd_dataset(args: argparse.Namespace) -> int:
    q = getattr(args, "query", None) or "protein dataset"
    n = getattr(args, "max_results", 5)
    src = getattr(args, "source", "github,hugging_face")
    return _run_script("src/tools/search/source/dataset_search.py", ["--query", q, "--max_results", str(n), "--source", src])


def cmd_web_search(args: argparse.Namespace) -> int:
    q = getattr(args, "query", None) or "protein language model"
    n = getattr(args, "max_results", 5)
    src = getattr(args, "source", "duckduckgo")
    return _run_script("src/tools/search/source/web_search.py", ["--query", q, "--max_results", str(n), "--source", src])


def cmd_foldseek(args: argparse.Namespace) -> int:
    pdb = getattr(args, "pdb_file", None)
    if not pdb:
        print("Error: --pdb_file required", file=sys.stderr)
        return 1
    out = getattr(args, "output_dir", "download/FoldSeek")
    start = getattr(args, "protect_start", 1)
    end = getattr(args, "protect_end", 10)
    return _run_script("src/tools/search/database/foldseek/__main__.py", [
        "--pdb_file", pdb, "--output_dir", out,
        "--protect_start", str(start), "--protect_end", str(end),
    ])


def cmd_uniprot(args: argparse.Namespace) -> int:
    uid = getattr(args, "uniprot_id", None)
    f = getattr(args, "file", None)
    if not uid and not f:
        print("Error: --uniprot_id or --file required", file=sys.stderr)
        return 1
    out = getattr(args, "out_dir", "download/uniprot_sequences") or "download/uniprot_sequences"
    cmd_args = ["-o", out]
    if uid:
        cmd_args = ["-i", uid] + cmd_args
    else:
        cmd_args = ["-f", f] + cmd_args
    if getattr(args, "merge", False):
        cmd_args.append("-m")
    if getattr(args, "num_workers", None) is not None:
        cmd_args += ["-n", str(args.num_workers)]
    if getattr(args, "error_file", None):
        cmd_args += ["-e", args.error_file]
    return _run_script("src/tools/search/database/uniprot/uniprot_sequence.py", cmd_args)


def cmd_ncbi(args: argparse.Namespace) -> int:
    i = getattr(args, "ncbi_id", None)
    f = getattr(args, "file", None)
    if not i and not f:
        print("Error: --id or --file required", file=sys.stderr)
        return 1
    out = getattr(args, "out_dir", None)
    if not out:
        print("Error: --out_dir required", file=sys.stderr)
        return 1
    cmd_args = ["-o", out, "-d", getattr(args, "db", "protein")]
    if i:
        cmd_args = ["-i", i] + cmd_args
    else:
        cmd_args = ["-f", f] + cmd_args
    if getattr(args, "merge", False):
        cmd_args.append("-m")
    if getattr(args, "num_workers", None) is not None:
        cmd_args += ["-n", str(args.num_workers)]
    if getattr(args, "error_file", None):
        cmd_args += ["-e", args.error_file]
    return _run_script("src/tools/search/database/ncbi/ncbi_sequence.py", cmd_args)


def cmd_alphafold(args: argparse.Namespace) -> int:
    uid = getattr(args, "uniprot_id", None)
    f = getattr(args, "uniprot_id_file", None)
    if not uid and not f:
        print("Error: --uniprot_id or --uniprot_id_file required", file=sys.stderr)
        return 1
    out = getattr(args, "out_dir", "download/alphafold2_structures")
    cmd_args = ["-o", out]
    if uid:
        cmd_args = ["-i", uid] + cmd_args
    else:
        cmd_args = ["-f", f] + cmd_args
    if getattr(args, "error_file", None):
        cmd_args += ["-e", args.error_file]
    if getattr(args, "num_workers", None) is not None:
        cmd_args += ["-n", str(args.num_workers)]
    return _run_script("src/tools/search/database/alphafold/alphafold_structure.py", cmd_args)


def cmd_rcsb(args: argparse.Namespace) -> int:
    pdb_id = getattr(args, "pdb_id", None)
    f = getattr(args, "pdb_id_file", None)
    if not pdb_id and not f:
        print("Error: --pdb_id or --pdb_id_file required", file=sys.stderr)
        return 1
    out = getattr(args, "out_dir", "download/rcsb_structures")
    fmt = getattr(args, "type", "pdb")
    cmd_args = ["-o", out, "-t", fmt]
    if getattr(args, "unzip", False):
        cmd_args.append("-u")
    if pdb_id:
        cmd_args = ["-i", pdb_id] + cmd_args
    else:
        cmd_args = ["-f", f] + cmd_args
    if getattr(args, "error_file", None):
        cmd_args += ["-e", args.error_file]
    if getattr(args, "num_workers", None) is not None:
        cmd_args += ["-n", str(args.num_workers)]
    return _run_script("src/tools/search/database/rcsb/rcsb_structure.py", cmd_args)


def cmd_interpro(args: argparse.Namespace) -> int:
    """Download InterPro entry metadata by InterPro ID(s). Use -i for single ID, -f for txt file."""
    i = getattr(args, "interpro_id", None)
    f = getattr(args, "interpro_id_file", None)
    if not i and not f:
        print("Error: --interpro_id or --interpro_id_file required", file=sys.stderr)
        return 1
    out = getattr(args, "out_dir", "download/interpro_metadata")
    cmd_args = ["-o", out]
    if i:
        cmd_args = ["-i", i] + cmd_args
    else:
        cmd_args = ["-f", f] + cmd_args
    if getattr(args, "error_file", None):
        cmd_args += ["-e", args.error_file]
    if getattr(args, "num_workers", None) is not None:
        cmd_args += ["-n", str(args.num_workers)]
    return _run_script("src/tools/search/database/interpro/interpro_metadata.py", cmd_args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Search tools CLI: literature, dataset, web search, FoldSeek, UniProt/NCBI/AF/RCSB/InterPro download.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # literature
    p = sub.add_parser("literature", help="Literature search (arXiv, PubMed, bioRxiv, Semantic Scholar)")
    p.add_argument("--query", default="protein structure prediction", help="Search query")
    p.add_argument("--max_results", type=int, default=5, help="Max results per source")
    p.add_argument("--source", default="arxiv,pubmed", help="Comma-separated: arxiv,pubmed,biorxiv,semantic_scholar")
    p.set_defaults(run=cmd_literature)

    # dataset
    p = sub.add_parser("dataset", help="Dataset search (GitHub, Hugging Face)")
    p.add_argument("--query", default="protein dataset", help="Search query")
    p.add_argument("--max_results", type=int, default=5, help="Max results per source")
    p.add_argument("--source", default="github,hugging_face", help="github and/or hugging_face")
    p.set_defaults(run=cmd_dataset)

    # web-search
    p = sub.add_parser("web-search", help="Web search (DuckDuckGo, Tavily)")
    p.add_argument("--query", default="protein language model", help="Search query")
    p.add_argument("--max_results", type=int, default=5, help="Max results")
    p.add_argument("--source", default="duckduckgo", help="duckduckgo and/or tavily")
    p.set_defaults(run=cmd_web_search)

    # foldseek
    p = sub.add_parser("foldseek", help="FoldSeek structure search: PDB -> alignments -> FASTA")
    p.add_argument("--pdb_file", required=True, help="Input PDB file path")
    p.add_argument("--output_dir", default="download/FoldSeek", help="Output directory")
    p.add_argument("--protect_start", type=int, default=1, help="Protected region start (1-based)")
    p.add_argument("--protect_end", type=int, default=10, help="Protected region end (1-based)")
    p.set_defaults(run=cmd_foldseek)

    # uniprot
    p = sub.add_parser("uniprot", help="Download FASTA from UniProt by ID(s)")
    p.add_argument("--uniprot_id", help="Single UniProt ID")
    p.add_argument("--file", help="File with one UniProt ID per line")
    p.add_argument("--out_dir", default="download/uniprot_sequences", help="Output directory")
    p.add_argument("--merge", action="store_true", help="Merge into one FASTA")
    p.add_argument("--num_workers", type=int, default=12, help="Parallel workers")
    p.add_argument("--error_file", help="File to log failed IDs")
    p.set_defaults(run=cmd_uniprot)

    # ncbi
    p = sub.add_parser("ncbi", help="Download FASTA from NCBI by accession ID(s)")
    p.add_argument("--id", dest="ncbi_id", help="Single NCBI accession ID")
    p.add_argument("--file", help="File with one NCBI ID per line")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--db", default="protein", help="Database: protein, nuccore")
    p.add_argument("--merge", action="store_true", help="Merge into one FASTA")
    p.add_argument("--num_workers", type=int, default=10, help="Parallel workers")
    p.add_argument("--error_file", help="File to log failed IDs")
    p.set_defaults(run=cmd_ncbi)

    # alphafold
    p = sub.add_parser("alphafold", help="Download AlphaFold structure(s) by UniProt ID(s)")
    p.add_argument("--uniprot_id", help="Single UniProt ID")
    p.add_argument("--uniprot_id_file", help="File with one UniProt ID per line")
    p.add_argument("--out_dir", default="download/alphafold2_structures", help="Output directory")
    p.add_argument("--error_file", help="File to log failed IDs")
    p.add_argument("--num_workers", type=int, default=12, help="Parallel workers")
    p.set_defaults(run=cmd_alphafold)

    # rcsb
    p = sub.add_parser("rcsb", help="Download PDB/CIF from RCSB by PDB ID(s)")
    p.add_argument("--pdb_id", help="Single PDB ID")
    p.add_argument("--pdb_id_file", help="File with one PDB ID per line")
    p.add_argument("--out_dir", default="download/rcsb_structures", help="Output directory")
    p.add_argument("--type", default="pdb", choices=["cif", "pdb", "pdb1", "xml", "sf", "mr", "mrstr"], help="File type")
    p.add_argument("--unzip", action="store_true", help="Unzip .gz after download")
    p.add_argument("--error_file", help="File to log failed IDs")
    p.add_argument("--num_workers", type=int, default=12, help="Parallel workers")
    p.set_defaults(run=cmd_rcsb)

    # interpro
    p = sub.add_parser("interpro", help="Download InterPro entry metadata by InterPro ID(s)")
    p.add_argument("-i", "--interpro_id", help="Single InterPro ID")
    p.add_argument("-f", "--interpro_id_file", help="Txt file with one InterPro ID per line")
    p.add_argument("-o", "--out_dir", default="download/interpro_metadata", help="Output directory")
    p.add_argument("-e", "--error_file", help="File to log failed IDs")
    p.add_argument("-n", "--num_workers", type=int, default=12, help="Parallel workers")
    p.set_defaults(run=cmd_interpro)

    args = parser.parse_args()
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
