import sys
import os
sys.path.append(os.getcwd())
import time
import argparse
import torch
import numpy as np
import datetime
import pandas as pd
from tqdm import tqdm
from Bio.PDB import PDBParser, MMCIFParser
from transformers import EsmTokenizer, EsmForMaskedLM
from src.mutation.utils import generate_mutations_from_sequence
from src.mutation.models.esm.inverse_folding.util import extract_seq_from_pdb
from typing import List

def extract_plddt(pdb_path: str) -> np.ndarray:
    """
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """

    # Initialize parser
    if pdb_path.endswith(".cif"):
        parser = MMCIFParser()
    elif pdb_path.endswith(".pdb"):
        parser = PDBParser()
    else:
        raise ValueError("Invalid file format for plddt extraction. Must be '.cif' or '.pdb'.")
    
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    chain = model["A"]

    # Extract plddt scores
    plddts = []
    for residue in chain:
        residue_plddts = []
        for atom in residue:
            plddt = atom.get_bfactor()
            residue_plddts.append(plddt)
        
        plddts.append(np.mean(residue_plddts))

    plddts = np.array(plddts)
    return plddts

def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = "auto",
                  plddt_threshold: float = 70.,
                  foldseek_verbose: bool = False) -> dict:
    """

    Args:
        foldseek: Binary executable file of foldseek

        path: Path to pdb file

        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.

        process_id: Process ID for temporary files. This is used for parallel processing.

        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.

        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

        foldseek_verbose: If True, foldseek will print verbose messages.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    # Check whether the structure is predicted by AlphaFold2
    if plddt_mask == "auto":
        with open(path, "r") as r:
            plddt_mask = True if "alphafold" in r.read().lower() else False
    
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_mask:
                try:
                    plddts = extract_plddt(path)
                    assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                    
                    # Mask regions with plddt < threshold
                    indices = np.where(plddts < plddt_threshold)[0]
                    np_seq = np.array(list(struc_seq))
                    np_seq[indices] = "#"
                    struc_seq = "".join(np_seq)
                
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Failed to mask plddt for {name}")
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict

def saprot_score(pdb_file: str, mutants: List[str], chain: str = "A", 
                 foldseek_path: str = None) -> List[float]:
    """
    Calculate SaProt scores for a list of mutations.
    
    Args:
        pdb_file: Path to the PDB file
        mutants: List of mutation strings (e.g., ["A1B", "C2D"])
        chain: Chain ID to extract from PDB
        foldseek_path: Path to foldseek binary (optional, will download if None)
        
    Returns:
        List of scores corresponding to the input mutations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load SaProt model and tokenizer
    model_path = "westlake-repl/SaProt_650M_AF2"
    tokenizer = EsmTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = EsmForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)

    foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"
    # Setup foldseek path
    if foldseek_path is None:
        foldseek_path = os.path.expanduser("~/.cache/huggingface/hub/models--westlake-repl--SaProt_650M_AF2/foldseek")
        if not os.path.exists(foldseek_path + "/foldseek"):
            os.system(f"mkdir -p {foldseek_path}")
            os.system(f"wget https://huggingface.co/tyang816/Foldseek_bin/resolve/main/foldseek -P {foldseek_path}")
            os.system(f"chmod +x {foldseek_path}/foldseek")
        else:
            print(f"Foldseek already exists at {foldseek_path}/foldseek")
    
    # Extract structural sequence
    parsed_seqs = get_struc_seq(foldseek_path + "/foldseek", pdb_file, [chain])[chain]
    seq, foldseek_seq, combined_seq = parsed_seqs
    
    # Tokenize the combined sequence (structure-aware sequence)
    inputs = tokenizer(combined_seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compute logits
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze()

    # Calculate scores for each mutation
    pred_scores = []
    for mutant in tqdm(mutants):
        mutant_score = 0
        sep = ":" if ":" in mutant else ";"
        for sub_mutant in mutant.split(sep):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]), sub_mutant[-1]
            ori_st = tokenizer.get_vocab()[wt + foldseek_struc_vocab[0]]
            mut_st = tokenizer.get_vocab()[mt + foldseek_struc_vocab[0]]
            
            ori_prob = logits[idx, ori_st: ori_st + len(foldseek_struc_vocab)].sum()
            mut_prob = logits[idx, mut_st: mut_st + len(foldseek_struc_vocab)].sum()

            pred = mut_prob - ori_prob
            mutant_score += pred.item()
        pred_scores.append(mutant_score / len(mutant.split(sep)))

    return pred_scores

def main():
    parser = argparse.ArgumentParser(description='saprot')
    parser.add_argument('--pdb_file', type=str, required=True, help='Path to the pdb file')
    parser.add_argument('--mutations_csv', type=str, default=None, help='Path to the mutations CSV file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to the output CSV file')
    parser.add_argument('--foldseek_path', type=str, default=None, required=False, help='Path to the foldseek binary')
    parser.add_argument('--chain', type=str, default="A", help='Chain to be processed')
    args = parser.parse_args()

    # Extract sequence from PDB for mutation generation
    seq = extract_seq_from_pdb(args.pdb_file)

    # Handle mutations
    if args.mutations_csv is not None:
        df = pd.read_csv(args.mutations_csv)
        mutants = df['mutant'].tolist()
    else:
        mutants = generate_mutations_from_sequence(seq)
        df = pd.DataFrame(mutants, columns=['mutant'])

    # Calculate scores using the new function
    scores = saprot_score(args.pdb_file, mutants, args.chain, args.foldseek_path)
    df['saprot_score'] = scores
    df = df.sort_values(by='saprot_score', ascending=False)
    
    # Save results
    if args.output_csv is not None:
        output_dir = os.path.dirname(args.output_csv)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
    else:
        file_name = f"{args.pdb_file.split('/')[-1].split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)
        print(f"Results saved to {file_name}")

if __name__ == "__main__":
    main()