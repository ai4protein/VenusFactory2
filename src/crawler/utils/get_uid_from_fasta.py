import argparse


"""
get uniprot ids from multi fasta file
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--multi_fasta_file", type=str, default=None)
    parser.add_argument("-u", "--uid_file", type=str, default=None)
    args = parser.parse_args()
    
    data = []
    with open(args.multi_fasta_file, "r") as f:
        for line in f:
            if line.startswith(">"):
                uid = line.split("|")[1]
                data.append(uid)
    with open(args.uid_file, "w") as f:
        f.write("\n".join(data))