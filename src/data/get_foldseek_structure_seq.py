import os
import argparse
import json
from tqdm import tqdm

# conda install -c conda-forge -c bioconda foldseek
def get_foldseek_structure_seq(pdb_dir, rm_tmp=True):
    # foldseek createdb INPUT_dir_with_structures tmp_db
    # foldseek lndb tmp_db_h tmp_db_ss_h
    # foldseek convert2fasta tmp_db_ss OUTPUT_3di.fasta
    # use command to generate foldseek structure seq
    os.makedirs("tmp_db", exist_ok=True)
    os.system(f"foldseek createdb {pdb_dir} tmp_db/tmp_db")
    os.system(f"foldseek lndb tmp_db/tmp_db_h tmp_db/tmp_db_ss_h")
    os.system(f"foldseek convert2fasta tmp_db/tmp_db_ss tmp_db/tmp_db_ss.fasta")
    
    results = []
    # read fasta file
    with open("tmp_db/tmp_db_ss.fasta", "r") as f:
        for line in tqdm(f):
            if line.startswith(">"):
                name = line.split()[0][1:]
                seq = next(f).strip()
                results.append({"name":name+'.pdb', "foldseek_seq":seq})
    
    if rm_tmp:
        os.system("rm -rf tmp_db")
        
    return results
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, default=None)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--rm_tmp", type=bool, default=True)
    args = parser.parse_args()
    
    results = get_foldseek_structure_seq(args.pdb_dir, args.rm_tmp)
    with open(args.out_file, "w") as f:
        f.write("\n".join([json.dumps(r) for r in results]))