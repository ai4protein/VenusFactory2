import argparse
import os
from utils import read_multi_fasta

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--fasta_file", type=str, default=None)
    parser.add_argument("-d", "--af_dir", type=str, default=None)
    parser.add_argument("-o", "--output_dir", type=str, default=None)
    parser.add_argument("-c", "--chunk_size", type=int, default=50)
    args = parser.parse_args()
    
    for f in os.listdir(args.af_dir):
        if "model" in f:
            os.rename(os.path.join(args.af_dir, f), os.path.join(args.af_dir, f"{f.split('-')[1]}.pdb"))
    
    downloaded_uids = [p[:-4] for p in os.listdir(args.af_dir)]
    seqs = read_multi_fasta(args.fasta_file)
    unfold_seqs = {}
    for head, seq in seqs.items():
        uid = head.split("|")[1].strip()
        if uid not in downloaded_uids:
            unfold_seqs[head] = seq
    total_seqs = len(unfold_seqs)
    print(f"Total unfold {total_seqs} sequences")
    idx = 0
    for head, seq in unfold_seqs.items():
        uid = head.split("|")[1]
        chunk_idx = idx // args.chunk_size
        os.makedirs(os.path.join(args.output_dir, f"chunk_{chunk_idx}"), exist_ok=True)
        with open(os.path.join(args.output_dir, f"chunk_{chunk_idx}", f"{uid}.fasta"), "w") as f:
            f.write(f"{head}\n{seq}")
        idx += 1
    print("Done")
    