import os
import shutil
import argparse
import json
import pandas as pd
from tqdm import tqdm

# conda install -c conda-forge -c bioconda foldseek
def get_foldseek_structure_seq(pdb_dir, rm_tmp=True):
    # foldseek createdb INPUT_dir_with_structures tmp_db
    # foldseek lndb tmp_db_h tmp_db_ss_h
    # foldseek convert2fasta tmp_db_ss OUTPUT_3di.fasta
    # use command to generate foldseek structure seq; tmp under workdir/.cache
    cache_root = os.path.join(os.getcwd(), ".cache")
    tmp_db_dir = os.path.join(cache_root, "foldseek_tmp_db")
    os.makedirs(tmp_db_dir, exist_ok=True)
    os.system(f"foldseek createdb {pdb_dir} {tmp_db_dir}/tmp_db")
    os.system(f"foldseek lndb {tmp_db_dir}/tmp_db_h {tmp_db_dir}/tmp_db_ss_h")
    os.system(f"foldseek convert2fasta {tmp_db_dir}/tmp_db_ss {tmp_db_dir}/tmp_db_ss.fasta")

    results = []
    with open(os.path.join(tmp_db_dir, "tmp_db_ss.fasta"), "r") as f:
        for line in tqdm(f):
            if line.startswith(">"):
                name = line.split()[0][1:]
                seq = next(f).strip()
                results.append({"name": name.split('.')[0], "foldseek_seq": seq})

    if rm_tmp:
        shutil.rmtree(tmp_db_dir, ignore_errors=True)

    return results
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, default=None)
    parser.add_argument("--out_file", type=str, default=None)
    parser.add_argument("--rm_tmp", type=bool, default=True)
    parser.add_argument("--out_format", type=str, default="csv", choices=["csv", "json"])
    args = parser.parse_args()
    
    results = get_foldseek_structure_seq(args.pdb_dir, args.rm_tmp)
    if args.out_format == "csv":
        pd.DataFrame(results).to_csv(args.out_file, index=False)
    else:
        with open(args.out_file, "w") as f:
            f.write("\n".join([json.dumps(r) for r in results]))