import argparse
import os
import sys
import shutil
import warnings
warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())
from tqdm import tqdm
from Bio.PDB import PDBParser, PPBuilder
from src.data_collect.utils import unzip, ungzip, get_seq_from_pdb, get_seqs_from_pdb


def unzip_files(unzip_dir):
    files = os.listdir(unzip_dir)
    bar = tqdm(files)
    for file in bar:
        bar.set_postfix({"current": file})
        ungzip(os.path.join(unzip_dir, file), unzip_dir)


def is_apo(pdb_path):
    parser = PDBParser()
    structure = parser.get_structure('pdb', pdb_path)

    for model in structure:
        for chain in model:
            for residue in chain:
                hetero_flag = residue.id[0].strip()
                if hetero_flag != '':
                    return False
    return True


def process(args):
    if args.is_zip:
        assert args.raw_dir, "no raw_dir"
        unzip_files(args.raw_dir)
    
    pdbs = sorted(os.listdir(args.raw_dir))
    seq_pdb_dic = {}
    bar = tqdm(pdbs)
    for pdb in bar:
        bar.set_postfix_str(f"{pdb}")
        try:
            seq = get_seq_from_pdb(os.path.join(args.raw_dir, pdb))
        except:
            continue
        if seq in seq_pdb_dic.keys():
            if is_apo(os.path.join(args.raw_dir, pdb)):
                seq_pdb_dic[seq] = pdb
            else:
                continue
        seq_pdb_dic[seq] = pdb
    for seq, pdb in seq_pdb_dic.items():
        shutil.copyfile(os.path.join(args.raw_dir, pdb), os.path.join(args.unique_dir, pdb))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_zip", action="store_true")
    parser.add_argument("--raw_dir", type=str, default="data/MDH/af/raw")
    parser.add_argument("--unique_dir", type=str, default="data/MDH/af/unique")
    
    args = parser.parse_args()
    process(args)
    # get_seqs_from_pdb("data/MDH/pdb/unique", "data/MDH/pdb/unique_seqs.fasta")
    
    