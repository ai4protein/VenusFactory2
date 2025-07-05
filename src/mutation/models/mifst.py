import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
import pandas as pd
from src.mutation.models.sequence_models.pdb_utils import parse_PDB, process_coords
from src.mutation.models.sequence_models.pretrained import load_model_and_alphabet
from src.mutation.models.sequence_models.constants import PROTEIN_ALPHABET
from src.mutation.utils import generate_mutations_from_sequence
from typing import List
from tqdm import tqdm

def mifst_score(pdb_file: str, mutants: List[str], model_location: str = 'mifst') -> List[float]:
    """
    Calculate MIF-ST scores for a list of mutations.
    
    Args:
        pdb_file: Path to the PDB file
        mutants: List of mutation strings (e.g., ["A1B", "C2D"])
        model_location: Path or name of the MIF-ST model
        
    Returns:
        List of scores corresponding to the input mutations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load MIF-ST model
    model, collater = load_model_and_alphabet(model_location)
    model = model.to(device)
    model.eval()

    # Parse PDB file and extract coordinates
    coords, sequence, _ = parse_PDB(pdb_file)
    coords = {
        'N': coords[:, 0],
        'CA': coords[:, 1],
        'C': coords[:, 2]
    }
    
    # Process coordinates
    dist, omega, theta, phi = process_coords(coords)
    batch = [[sequence, torch.tensor(dist, dtype=torch.float).to(device),
              torch.tensor(omega, dtype=torch.float).to(device),
              torch.tensor(theta, dtype=torch.float).to(device),
              torch.tensor(phi, dtype=torch.float).to(device)]]
    
    # Prepare input for model
    src, nodes, edges, connections, edge_mask = collater(batch)
    
    # Compute logits
    with torch.no_grad():
        logits = model(src.to(device), nodes.to(device), edges.to(device), 
                      connections.to(device), edge_mask.to(device), result='logits')
    # logits shape: (1, seq_len, 20)

    # Calculate scores for each mutation
    pred_scores = []
    for mutant in tqdm(mutants):
        mutant_score = 0
        sep = ":" if ":" in mutant else ";"
        for sub_mut in mutant.split(sep):
            if sub_mut.lower() == "wt":
                continue
            wt, idx, mt = sub_mut[0], int(sub_mut[1:-1]) - 1, sub_mut[-1]
            wt_encoded, mt_encoded = PROTEIN_ALPHABET.index(wt), PROTEIN_ALPHABET.index(mt)
            pred = logits[0, idx, mt_encoded] - logits[0, idx, wt_encoded]
            mutant_score += pred.item()
        pred_scores.append(mutant_score / len(mutant.split(sep)))

    return pred_scores


def main():
    parser = argparse.ArgumentParser(description='MIF-ST')
    parser.add_argument('--pdb_file', type=str, required=True, help='Path to the pdb file')
    parser.add_argument('--mutations_csv', type=str, default=None, help='Path to the mutations CSV file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to the output CSV file')
    parser.add_argument('--model_location', type=str, default='mifst', help='Path or name of the MIF-ST model')
    args = parser.parse_args()

    # Parse PDB file to get sequence
    coords, sequence, _ = parse_PDB(args.pdb_file)

    # Handle mutations
    if args.mutations_csv is not None:
        df = pd.read_csv(args.mutations_csv)
        mutants = df['mutant'].tolist()
    else:
        mutants = generate_mutations_from_sequence(sequence)
        df = pd.DataFrame(mutants, columns=['mutant'])

    # Calculate scores using the new function
    scores = mifst_score(args.pdb_file, mutants, args.model_location)
    df['mifst_score'] = scores

    # Save results
    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
    else:
        import datetime
        file_name = f"{os.path.splitext(os.path.basename(args.pdb_file))[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)

if __name__ == "__main__":
    main()