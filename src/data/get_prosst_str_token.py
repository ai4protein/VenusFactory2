import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import argparse
import json
import pandas as pd
import torch
from src.data.prosst.structure.get_sst_seq import SSTPredictor
import warnings
warnings.filterwarnings("ignore", category=Warning)


def convert_predict_results(predict_results, structure_vocab_size):
    """Convert SSTPredictor.predict_from_pdb output to name/aa_seq/struct_tokens format."""
    out = []
    key_sst = f"{structure_vocab_size}_sst_seq"
    for r in predict_results:
        if "error" in r:
            continue
        name = r.get("name", "").split('.')[0]  # Same as other get_seq: keep name without .ef/.pdb
        aa_seq = r.get("aa_seq", "")
        sst_seq = r.get(key_sst, [])
        sst_seq = [str(i + 3) for i in sst_seq]
        out.append({
            "name": name,
            "aa_seq": aa_seq,
            f"{structure_vocab_size}_struct_tokens": sst_seq,
        })
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ProSST structure token generator')
    parser.add_argument('--pdb_dir', type=str, help='Directory containing PDB files')
    parser.add_argument('--pdb_file', type=str, help='Single PDB file path')
    parser.add_argument('--structure_vocab_size', type=int, default=2048, help='Structure vocab size (20, 64, 128, 512, 1024, 2048, 4096)')
    # SSTPredictor init
    parser.add_argument('--num_processes', type=int, default=12, help='Number of processes for subgraph building')
    parser.add_argument('--num_threads', type=int, default=16, help='Number of threads for subgraph building')
    parser.add_argument('--max_batch_nodes', type=int, default=10000, help='Max nodes per batch')
    parser.add_argument('--max_distance', type=float, default=10, help='Max distance for edges')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu), default auto')
    parser.add_argument('--cache_subgraph_dir', type=str, default=None, help='Directory to cache subgraphs')
    # I/O
    parser.add_argument('--pdb_index_file', type=str, default=None, help='PDB index file for sharding')
    parser.add_argument('--pdb_index_level', type=int, default=1, help='Directory hierarchy depth')
    parser.add_argument('--error_file', type=str, default=None, help='Error log output path')
    parser.add_argument('--out_file', type=str, required=True, help='Output file path')
    parser.add_argument('--out_format', type=str, default='csv', choices=['csv', 'json'])
    args = parser.parse_args()

    if args.pdb_dir is not None:
        if args.pdb_index_file:
            pdbs = open(args.pdb_index_file).read().splitlines()
            pdb_files = []
            for pdb in pdbs:
                pdb = pdb.strip()
                pdb_relative_dir = args.pdb_dir
                for i in range(1, args.pdb_index_level + 1):
                    pdb_relative_dir = os.path.join(pdb_relative_dir, pdb[:i])
                pdb_files.append(os.path.join(pdb_relative_dir, pdb + ".pdb"))
        else:
            pdb_files = sorted([
                os.path.join(args.pdb_dir, p)
                for p in os.listdir(args.pdb_dir)
                if p.endswith('.pdb') or p.endswith('.ef.pdb')
            ])

        if not pdb_files:
            raise FileNotFoundError(f"No PDB files found in {args.pdb_dir}")

        processor = SSTPredictor(
            structure_vocab_size=args.structure_vocab_size,
            num_processes=args.num_processes,
            num_threads=args.num_threads,
            max_batch_nodes=args.max_batch_nodes,
            max_distance=args.max_distance,
            device=args.device,
        )
        predict_results = processor.predict_from_pdb(
            pdb_files,
            error_file=args.error_file,
            cache_subgraph_dir=args.cache_subgraph_dir,
        )
        results = convert_predict_results(predict_results, args.structure_vocab_size)

        if args.out_format == 'csv':
            df = pd.DataFrame(results)
            for col in df.columns:
                if col != 'name' and col != 'aa_seq' and df[col].dtype == object:
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
            df.to_csv(args.out_file, index=False)
        else:
            with open(args.out_file, 'w') as f:
                f.write('\n'.join(json.dumps(r) for r in results))

    elif args.pdb_file:
        processor = SSTPredictor(
            structure_vocab_size=args.structure_vocab_size,
            num_processes=args.num_processes,
            num_threads=args.num_threads,
            max_batch_nodes=args.max_batch_nodes,
            max_distance=args.max_distance,
            device=args.device,
        )
        predict_results = processor.predict_from_pdb([args.pdb_file], error_file=args.error_file)
        results = convert_predict_results(predict_results, args.structure_vocab_size)
        if not results:
            raise RuntimeError(f"predict_from_pdb returned no valid result for {args.pdb_file}")
        result = results[0]

        if args.out_format == 'csv':
            df = pd.DataFrame([result])
            for col in df.columns:
                if col != 'name' and col != 'aa_seq' and isinstance(df[col].iloc[0], list):
                    df[col] = df[col].apply(json.dumps)
            df.to_csv(args.out_file, index=False)
        else:
            with open(args.out_file, 'w') as f:
                json.dump(result, f)
