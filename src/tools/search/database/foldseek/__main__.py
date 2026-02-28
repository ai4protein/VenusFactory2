"""CLI entry for FoldSeek: python -m tools.search.database.foldseek --pdb_file <path> ..."""
import argparse

from . import get_foldseek_sequences


def main():
    parser = argparse.ArgumentParser(description="FoldSeek structure search: PDB -> alignments -> FASTA")
    parser.add_argument("--pdb_file", type=str, required=True, help="Input PDB file path")
    parser.add_argument("--output_dir", type=str, default="download/FoldSeek", help="Output directory")
    parser.add_argument("--protect_start", type=int, default=1, help="Protected region start (1-based)")
    parser.add_argument("--protect_end", type=int, default=10, help="Protected region end (1-based)")
    args = parser.parse_args()
    foldseek_fasta, total_sequences = get_foldseek_sequences(
        args.pdb_file, args.protect_start, args.protect_end, output_dir=args.output_dir
    )
    print(f"Success! Fasta: {foldseek_fasta}, sequences: {total_sequences}")


if __name__ == "__main__":
    main()
