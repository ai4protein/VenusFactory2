import os
import argparse
import json
import torch
from src.data.prosst.structure.get_sst_seq import SSTPredictor
from src.utils.data_utils import extract_seq_from_pdb
import warnings
warnings.filterwarnings("ignore", category=Warning)


def get_prosst_token(pdb_file, processor, structure_vocab_size):
    """Generate ProSST structure tokens for a PDB file"""
    try:
        aa_seq = extract_seq_from_pdb(pdb_file)
        structure_result = processor(pdb_file)
        pdb_name = os.path.basename(pdb_file)

        sst_seq = structure_result[f'{structure_vocab_size}_sst_seq']
        sst_seq = [str(i+3) for i in sst_seq]

        return {
            "name": os.path.basename(pdb_file).split('.')[0],
            "aa_seq": aa_seq,
            f"{structure_vocab_size}_struct_tokens": sst_seq
        }, None
        
    except Exception as e:
        return pdb_file, f"{str(e)}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ProSST structure token generator')
    parser.add_argument('--pdb_dir', type=str, help='Directory containing PDB files')
    parser.add_argument('--pdb_file', type=str, help='Single PDB file path')
    parser.add_argument('--structure_vocab_size', type=int, default=[20], nargs='+', help='structure vocab size')
    parser.add_argument('--num_workers', type=int, default=16, help='Number of parallel workers')
    parser.add_argument('--pdb_index_file', type=str, default=None, help='PDB index file for sharding')
    parser.add_argument('--pdb_index_level', type=int, default=1, help='Directory hierarchy depth')
    parser.add_argument('--error_file', type=str, help='Error log output path')
    parser.add_argument('--out_file', type=str, required=True, help='Output JSON file path')
    args = parser.parse_args()

    if args.pdb_dir is not None:
        # load pdb index file
        if args.pdb_index_file:            
            pdbs = open(args.pdb_index_file).read().splitlines()
            pdb_files = []
            for pdb in pdbs:
                pdb_relative_dir = args.pdb_dir
                for i in range(1, args.pdb_index_level+1):
                    pdb_relative_dir = os.path.join(pdb_relative_dir, pdb[:i])
                pdb_files.append(os.path.join(pdb_relative_dir, pdb+".pdb"))
        
        # regular pdb dir
        else:
            pdb_files = sorted([os.path.join(args.pdb_dir, p) for p in os.listdir(args.pdb_dir)])
           
        results, errors = [], []
        for v in args.structure_vocab_size:
            processor = SSTPredictor(structure_vocab_size=v)
            results, errors = get_prosst_token(pdb_files, processor)


        with open(args.out_file, 'w') as f:
            f.write('\n'.join(json.dumps(r) for r in results))


    elif args.pdb_file:
        result, error = get_prosst_token(args.pdb_file)
        if error:
            raise RuntimeError(f"Error processing {args.pdb_file}: {error}")
        with open(args.out_file, 'w') as f:
            json.dump(result, f)