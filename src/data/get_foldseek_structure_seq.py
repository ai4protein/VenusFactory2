import os
import sys
import json
from tqdm import tqdm

# conda install -c conda-forge -c bioconda foldseek
def get_struc_seq(pdb_dir, out_file):
    # foldseek createdb INPUT_dir_with_structures tmp_db
    # foldseek lndb tmp_db_h tmp_db_ss_h
    # foldseek convert2fasta tmp_db_ss OUTPUT_3di.fasta
    # use command to generate foldseek structure seq
    os.makedirs("tmp_db", exist_ok=True)
    os.system(f"foldseek createdb {pdb_dir} tmp_db/tmp_db")
    os.system(f"foldseek lndb tmp_db/tmp_db_h tmp_db/tmp_db_ss_h")
    os.system(f"foldseek convert2fasta tmp_db/tmp_db_ss tmp_db/tmp_db_ss.fasta")
    
    final_data = []
    # read fasta file
    with open("tmp_db/tmp_db_ss.fasta", "r") as f:
        for line in tqdm(f):
            if line.startswith(">"):
                name = line.split()[0][1:]
                seq = next(f).strip()
                data_line = {"name":name+'.pdb', "foldseek_seq":seq}
                final_data.append(data_line)
                
    with open(out_file, "w") as f:
        f.write("\n".join([json.dumps(r) for r in final_data]))
    
if __name__ == '__main__':
    get_struc_seq("dataset/sesadapter/DeepET/esmfold_pdb", "dataset/sesadapter/DeepET/esmfold_foldseek.json")