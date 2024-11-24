import zipfile
import gzip
import shutil
import os
from tqdm import tqdm
from Bio.PDB import PDBParser, PPBuilder

def unzip(zipath, savefolder):
    zf = zipfile.ZipFile(zipath)
    zf.extractall(savefolder)
    zf.close()
 
def ungzip(file, out_dir):
    with gzip.open(file, 'rb') as f_in:
        with open(os.path.join(out_dir, file.split('/')[-1][:-3]), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def get_seq_from_pdb(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure(pdb_file[-8:-4], pdb_file)
    ppb = PPBuilder()
    chain = structure[0]['A']
    seq = ""
    for pp in ppb.build_peptides(chain):
        seq += pp.get_sequence()
    return seq


def get_seqs_from_pdb(pdb_dir, out_file_path):
    pdbs = os.listdir(pdb_dir)
    with open(out_file_path, "w") as f:
        for pdb in tqdm(pdbs):
            seq = get_seq_from_pdb(os.path.join(pdb_dir, pdb))
            f.write(f"> {pdb}\n{seq}\n")
            
def read_multi_fasta(file_path):
    """
    params:
        file_path: path to a fasta file
    return:
        a dictionary of sequences
    """
    sequences = {}
    current_sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences[header] = current_sequence
                    current_sequence = ''
                header = line
            else:
                current_sequence += line
        if current_sequence:
            sequences[header] = current_sequence
    return sequences

def make_uid_chunks(uid_file, chunk_dir=None, chunk_size=10000):
    """
    params: 
        uid_file: path to a file containing a list of uniprot ids
        chunk_size: size of each chunk
    return:
        files containing chunks of uniprot ids
    """
    uids = [f.strip() for f in open(uid_file, "r").readlines()]
    uid_path = os.path.dirname(uid_file)
    if chunk_dir is None:
        chunk_dir = uid_path + "/chunks"
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_num = len(uids) // chunk_size + 1
    chunk_name = uid_file.split('/')[-1].split(".")[0]
    for i in range(chunk_num):
        with open(os.path.join(chunk_dir, f"{chunk_name}_{i}.txt"), "w") as f:
            f.write("\n".join(uids[i*chunk_size:(i+1)*chunk_size]))