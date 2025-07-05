import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
import numpy as np
import datetime
import pandas as pd
import warnings
import src.mutation.models.esm as esm
import torch.nn.functional as F
from src.mutation.models.esm.inverse_folding.util import CoordBatchConverter
from src.mutation.models.esm import pretrained
from tqdm import tqdm
from src.mutation.utils import generate_mutations_from_sequence
from typing import List

warnings.filterwarnings("ignore")

def full_sequence(origin_sequence, raw_mutant, offset):
    # if the mutant is in the format of "A123V;A124V"
    sep = ";"
    # if the mutant is in the format of "A123V:A124V"
    if ":" in raw_mutant:
        sep = ":"
        
    sequence = origin_sequence
    for raw_mut in raw_mutant.split(sep):
        # if the mutant is "WT"
        if raw_mut.lower() == "wt":
            continue
        to = raw_mut[-1]
        pos = int(raw_mut[1:-1]) - offset
        assert sequence[pos] == raw_mut[0], print(sequence[pos], raw_mut[0], pos)
        sequence = sequence[:pos] + to + sequence[pos + 1:]
    return sequence

def load_coords_and_sequence(pdb_path: str, chain: str = "A") -> tuple:
    """
    Load coordinates and sequence from PDB file.
    
    Args:
        pdb_path: Path to PDB file
        chain: Chain ID to extract
        
    Returns:
        tuple: (coordinates, sequence)
    """
    coords, pdb_seq = esm.inverse_folding.util.load_coords(pdb_path, chain)
    return coords, pdb_seq


def esmif1_score(pdb_file: str, mutants: List[str], chain: str = "A", 
                 model_name: str = "esm_if1_gvp4_t16_142M_UR50", 
                 exhaustive: bool = False) -> List[float]:
    """
    Calculate ESM-IF1 scores for a list of mutations.
    
    Args:
        pdb_file: Path to the PDB file
        mutants: List of mutation strings (e.g., ["A1B", "C2D"])
        chain: Chain ID to extract from PDB
        model_name: ESM-IF1 model name
        exhaustive: Whether to use exhaustive mode
        
    Returns:
        List of scores corresponding to the input mutations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading ESM-IF1 model: {model_name}")
    model, alphabet = pretrained.load_model_and_alphabet(model_name)
    model.eval()
    model = model.to(device)
    
    # Load coordinates and sequence
    print(f"Loading coordinates from: {pdb_file}")
    coords, pdb_seq = load_coords_and_sequence(pdb_file, chain)
    print(f"Sequence length: {len(pdb_seq)}")
    
    # Prepare batch converter
    batch_converter = CoordBatchConverter(alphabet)
    
    # Use cached computation for efficiency
    batch = [(coords, None, pdb_seq)]
    coords_, confidence, strs, tokens, padding_mask = batch_converter(batch)
    prev_output_tokens = tokens[:, :-1]
    logits, _ = model.forward(
        coords_.to(device),
        padding_mask.to(device),
        confidence.to(device),
        prev_output_tokens.to(device)
    )
    
    # Calculate scores for each mutation
    scores = []
    print(f"Processing {len(mutants)} mutations...")
    
    for mutation in tqdm(mutants):
        # Create mutated sequence
        mutated_seq = full_sequence(pdb_seq, mutation, 1)
        
        # Prepare batch for inference
        batch = [(coords, None, mutated_seq)]
        coords_, confidence, strs, tokens, padding_mask = batch_converter(batch)
        
        if exhaustive:
            # Forward pass
            with torch.no_grad():
                logits, _ = model.forward(
                    coords_.to(device),
                    padding_mask.to(device),
                    confidence.to(device),
                    tokens[:, :-1].to(device)  # prev_output_tokens
                )
        
        # Calculate loss
        target = tokens[:, 1:].to(device)
        target_padding_mask = (target == alphabet.padding_idx)
        
        loss = F.cross_entropy(logits, target, reduction='none')
        avg_loss = torch.sum(loss * ~target_padding_mask, dim=-1) / torch.sum(~target_padding_mask, dim=-1)
        
        # Convert to score (negative loss for higher probability = better score)
        score = -avg_loss.detach().cpu().numpy().item()
        scores.append(score)
    
    return scores


def main():
    parser = argparse.ArgumentParser(description='ESM-IF1 protein mutation scoring')
    parser.add_argument('--pdb_file', type=str, required=True, help='Path to the PDB file')
    parser.add_argument('--mutations_csv', type=str, default=None, help='Path to the mutations CSV file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to the output CSV file')
    parser.add_argument('--chain', type=str, default="A", help='Chain to be processed')
    parser.add_argument('--exhaustive', action='store_true', help='Use exhaustive mode')
    args = parser.parse_args()

    # Load coordinates and sequence to get the sequence for mutation generation
    coords, pdb_seq = load_coords_and_sequence(args.pdb_file, args.chain)
    
    # Handle mutations
    if args.mutations_csv is not None:
        print(f"Loading mutations from: {args.mutations_csv}")
        df = pd.read_csv(args.mutations_csv)
        mutants = df['mutant'].tolist()
    else:
        print("Generating all possible single mutations...")
        mutants = generate_mutations_from_sequence(pdb_seq)
        df = pd.DataFrame(mutants, columns=['mutant'])
    
    # Calculate scores using the new function
    scores = esmif1_score(
        pdb_file=args.pdb_file,
        mutants=mutants,
        chain=args.chain,
        exhaustive=args.exhaustive
    )
    
    # Add scores to dataframe
    df['esmif1_score'] = scores
    
    # Sort by score (higher is better)
    df = df.sort_values('esmif1_score', ascending=False)
    
    # Save results
    if args.output_csv is not None:
        output_path = args.output_csv
    else:
        file_name = f"{os.path.basename(args.pdb_file).split('.')[0]}_esmif1_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        output_path = file_name
    
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main() 