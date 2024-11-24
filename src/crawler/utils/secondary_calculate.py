import argparse
import os
import sys
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP


def process(args):
    pdbs = sorted(os.listdir(args.pdb_dir))
    bar = tqdm(pdbs)
    wrong_pdb = []
    for pdb in bar:
        if os.path.exists(os.path.join(args.out_dir, pdb[:-4])):
            continue
        bar.set_postfix_str(f"{pdb}")
        file = os.path.join(args.pdb_dir, pdb)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("pdb", file)
        model = structure[0]
        try:
            dssp = DSSP(model, file)
            with open(os.path.join(args.out_dir, pdb[:-4]), "w") as f:
                ss = ""
                for residue in dssp:
                    ss += residue[2]
                ss = ss.replace("-", "L")
                f.write(ss)
        except:
            wrong_pdb.append(pdb)
    print(wrong_pdb)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, default="data/MDH/pdb/process/PDB")
    parser.add_argument("--out_dir", type=str, default="data/MDH/pdb/process/SS")
    
    args = parser.parse_args()
    
    process(args)