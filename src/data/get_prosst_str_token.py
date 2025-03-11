import os
import sys
import argparse
import json
import pandas as pd
import torch
from tqdm import tqdm
from Bio import SeqIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from prosst.structure.quantizer import PdbQuantizer
from data_utils import extract_seq_from_pdb
import warnings
warnings.filterwarnings("ignore", category=Warning)
structure_vocab_size = 20
processor = PdbQuantizer(structure_vocab_size = structure_vocab_size)

def get_prosst_token(pdb_file):
    """Generate ProSST structure tokens for a PDB file"""
    try:
        # 提取氨基酸序列
        aa_seq = extract_seq_from_pdb(pdb_file)
        
        # 处理结构序列
        structure_result = processor(pdb_file)
        pdb_name = os.path.basename(pdb_file)
        # 验证数据结构
        if structure_vocab_size not in structure_result:
            raise ValueError(f"Missing structure key: {structure_vocab_size}")
        if pdb_name not in structure_result[structure_vocab_size]:
            raise ValueError(f"Missing PDB entry: {pdb_name}")
        
        struct_sequence = structure_result[structure_vocab_size][pdb_name]['struct']
        struct_sequence = [int(num) for num in struct_sequence]
        
        # 添加特殊标记 [1] + sequence + [2]
        structure_sequence_offset = [3 + num for num in struct_sequence]
        structure_input_ids = torch.tensor(
            [[1] + structure_sequence_offset + [2]], 
            dtype=torch.long
        )
        
        return {
            "name": os.path.basename(pdb_file).split('.')[0],
            "aa_seq": aa_seq,
            "struct_tokens": structure_input_ids[0].tolist()
        }, None
        
    except Exception as e:
        return pdb_file, f"{str(e)}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ProSST structure token generator')
    parser.add_argument('--pdb_dir', type=str, help='Directory containing PDB files')
    parser.add_argument('--pdb_file', type=str, help='Single PDB file path')
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
           

        # 并行处理
        results, errors = [], []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(get_prosst_token, f): f for f in pdb_files}
            with tqdm(total=len(futures), desc="Processing PDBs") as progress:
                for future in as_completed(futures):
                    result, error = future.result()
                    if error:
                        errors.append({"file": result, "error": error})
                    else:
                        results.append(result)
                    progress.update(1)

        if errors:
            error_path = args.error_file or args.out_file.replace('.json', '_errors.csv')
            pd.DataFrame(errors).to_csv(error_path, index=False)
            print(f"Encountered {len(errors)} errors. Saved to {error_path}")


        with open(args.out_file, 'w') as f:
            f.write('\n'.join(json.dumps(r) for r in results))


    elif args.pdb_file:
        result, error = get_prosst_token(args.pdb_file)
        if error:
            raise RuntimeError(f"Error processing {args.pdb_file}: {error}")
        with open(args.out_file, 'w') as f:
            json.dump(result, f)