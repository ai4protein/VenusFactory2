import os
import argparse
import subprocess
import shutil
from tqdm import tqdm

"""
Install maxit first
https://sw-tools.rcsb.org/apps/MAXIT/index.html
"""

def convert(file, maxit_o=1, out_dir=None, postfix=None):
    converted_file = file[:-4] + postfix
    if out_dir:
        converted_file = os.path.join(out_dir, converted_file.split('/')[-1])
    subprocess.run(["maxit", "-input", file, "-output", converted_file, "-o", str(maxit_o)])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--strategy", type=str, choices=["pdb2cif", "cif2pdb", "cif2mmcif"], default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    
    if args.strategy == "pdb2cif":
        maxit_o = 1
        postfix = ".cif"
    elif args.strategy == "cif2pdb":
        maxit_o = 2
        postfix = ".pdb"
    elif args.strategy == "cif2mmcif":
        maxit_o = 8
        postfix = ".cif"
    
    if args.input_dir:
        for file in tqdm(os.listdir(args.input_dir)):
            args.file = os.path.join(args.input_dir, file)
            convert(args.file, maxit_o, args.out_dir, postfix)
    else:
        convert(args.file, maxit_o, args.out_dir, postfix)
