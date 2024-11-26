import os
import sys
sys.path.append(os.getcwd())
import argparse
import json
import pandas as pd
from tqdm import tqdm
from Bio import PDB
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from src.utils.data_utils import extract_seq_from_pdb


ss_alphabet = ['H', 'E', 'C']
ss_alphabet_dic = {
    "H": "H", "G": "H", "E": "E",
    "B": "E", "I": "C", "T": "C",
    "S": "C", "L": "C", "-": "C",
    "P": "C"
}

def generate_feature(pdb_file):
    try:
        # extract amino acid sequence
        aa_seq = extract_seq_from_pdb(pdb_file)
        pdb_parser = PDB.PDBParser(QUIET=True)
        structure = pdb_parser.get_structure("protein", pdb_file)
        model = structure[0]
        dssp = PDB.DSSP(model, pdb_file)
        # extract secondary structure sequence
        sec_structures = []
        for i, dssp_res in enumerate(dssp):
            sec_structures.append(dssp_res[2])
    
    except Exception as e:
        return pdb_file, e

    sec_structure_str_8 = ''.join(sec_structures)
    sec_structure_str_8 = sec_structure_str_8.replace('-', 'L')
    if len(aa_seq) != len(sec_structure_str_8):
        return pdb_file, f"aa_seq {len(aa_seq)} and sec_structure_str_8 {len(sec_structure_str_8)} length mismatch"
    
    sec_structure_str_3 = ''.join([ss_alphabet_dic[ss] for ss in sec_structures])
    
    final_feature = {}
    final_feature["name"] = pdb_file.split('/')[-1]
    final_feature["aa_seq"] = aa_seq
    final_feature["ss8_seq"] = sec_structure_str_8
    final_feature["ss3_seq"] = sec_structure_str_3

    return final_feature, None
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_dir', type=str, help='pdb dir')
    parser.add_argument('--pdb_file', type=str, help='pdb file')
    
    # multi processing
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    
    # index pdb for large scale inference
    parser.add_argument("--pdb_index_file", default=None, type=str, help="pdb index file")
    parser.add_argument("--pdb_index_level", default=1, type=int, help="pdb index level")
    
    # save file
    parser.add_argument('--error_file', type=str, help='save error file')
    parser.add_argument('--out_file', type=str, help='save file')
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.out_file)
    os.makedirs(out_dir, exist_ok=True)
    
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
            
        results, error_pdbs, error_messages = [], [], []
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(generate_feature, pdb_file) for pdb_file in pdb_files]

            with tqdm(total=len(pdb_files), desc="Processing pdb") as progress:
                for future in as_completed(futures):
                    result, message = future.result()
                    if message is None:
                        results.append(result)
                    else:
                        error_pdbs.append(result)
                        error_messages.append(message)
                    progress.update(1)
            progress.close()
        
        if error_pdbs:
            if args.error_file is None:
                args.error_file = args.out_file.split(".")[0]+"_error.csv"
            error_dir = os.path.dirname(args.error_file)
            os.makedirs(error_dir, exist_ok=True)
            error_info = {"error_pdbs": error_pdbs, "error_messages": error_messages}
            pd.DataFrame(error_info).to_csv(args.error_file, index=False)
        
        with open(args.out_file, "w") as f:
            f.write("\n".join([json.dumps(r) for r in results]))
    
    elif args.pdb_file is not None:
        result, message = generate_feature(args.pdb_file)
        with open(args.out_file, "w") as f:
            json.dump(result, f)
    