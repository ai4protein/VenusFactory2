import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
import datetime
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.data.prosst.structure.get_sst_seq import SSTPredictor
from src.mutation.utils import generate_mutations_from_sequence
from src.mutation.models.esm.inverse_folding.util import extract_seq_from_pdb
from typing import List
from tqdm import tqdm


def prosst_score(pdb_file: str, mutants: List[str]) -> List[float]:
    """
    Calculate ProSST scores for a list of mutations.
    
    Args:
        pdb_file: Path to the PDB file
        mutants: List of mutation strings (e.g., ["A1B", "C2D"])
        
    Returns:
        List of scores corresponding to the input mutations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ProSST model and tokenizer
    prosst_model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True).to(device)
    prosst_tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
    predictor = SSTPredictor(structure_vocab_size=2048)

    # Extract structure sequence from PDB
    structure_sequence = predictor.predict_from_pdb(pdb_file)[0]['2048_sst_seq']
    structure_sequence_offset = [i + 3 for i in structure_sequence]

    # Extract residue sequence from PDB
    residue_sequence = extract_seq_from_pdb(pdb_file)

    # Tokenize sequences
    tokenized_res = prosst_tokenizer([residue_sequence], return_tensors='pt')
    input_ids = tokenized_res['input_ids'].to(device)
    attention_mask = tokenized_res['attention_mask'].to(device)
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0).to(device)

    # Compute logits
    with torch.no_grad():
        outputs = prosst_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=structure_input_ids
        )
    logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()

    # Get vocabulary for scoring
    vocab = prosst_tokenizer.get_vocab()
    
    # Calculate scores for each mutation
    pred_scores = []
    for mutant in tqdm(mutants):
        mutant_score = 0
        sep = ":" if ":" in mutant else ";"
        for sub_mutant in mutant.split(sep):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            # Calculate log probability difference: log(P(mutant)) - log(P(wildtype))
            pred = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
            mutant_score += pred.item()
        pred_scores.append(mutant_score / len(mutant.split(sep)))

    return pred_scores


def main():
    parser = argparse.ArgumentParser(description='Prosst')
    parser.add_argument('--pdb_file', type=str, required=True, help='Path to the pdb file')
    parser.add_argument('--mutations_csv', type=str, default=None, help='Path to the mutations CSV file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to the output CSV file')
    args = parser.parse_args()

    # Extract residue sequence from PDB
    residue_sequence = extract_seq_from_pdb(args.pdb_file)

    # Handle mutations
    if args.mutations_csv is not None:
        df = pd.read_csv(args.mutations_csv)
        mutants = df['mutant'].tolist()
    else:
        mutants = generate_mutations_from_sequence(residue_sequence)
        df = pd.DataFrame(mutants, columns=['mutant'])

    # Calculate scores using the new function
    scores = prosst_score(args.pdb_file, mutants)
    df['prosst_score'] = scores

    # Save results
    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
    else:
        file_name = f"{args.pdb_file.split('/')[-1].split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)

if __name__ == "__main__":
    main()