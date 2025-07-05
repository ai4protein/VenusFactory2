import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
import datetime
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.mutation.utils import generate_mutations_from_sequence
from typing import List


def esm1v_score(fasta_file: str, mutants: List[str], 
                model_name: str = "facebook/esm1v_t33_650M_UR90S_1") -> List[float]:
    """
    Calculate ESM1V scores for a list of mutations.
    
    Args:
        fasta_file: Path to the FASTA file
        mutants: List of mutation strings (e.g., ["A1B", "C2D"])
        model_name: ESM1V model name
        
    Returns:
        List of scores corresponding to the input mutations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ESM1V model and tokenizer
    esm1v_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    esm1v_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load sequence from FASTA file
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        sequence = ''.join(line.strip() for line in lines[1:])

    # Tokenize sequence
    tokenized_res = esm1v_tokenizer([sequence], return_tensors='pt')
    vocab = esm1v_tokenizer.get_vocab()
    input_ids = tokenized_res['input_ids'].to(device)
    attention_mask = tokenized_res['attention_mask'].to(device)

    # Compute logits
    with torch.no_grad():
        outputs = esm1v_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()

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
    parser = argparse.ArgumentParser(description='ESM1V')
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to the fasta file')
    parser.add_argument('--mutations_csv', type=str, default=None, help='Path to the mutations CSV file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to the output CSV file')
    args = parser.parse_args()

    # Load sequence from FASTA file
    with open(args.fasta_file, 'r') as f:
        lines = f.readlines()
        sequence = ''.join(line.strip() for line in lines[1:])

    # Handle mutations
    if args.mutations_csv is not None:
        df = pd.read_csv(args.mutations_csv)
        mutants = df['mutant'].tolist()
    else:
        mutants = generate_mutations_from_sequence(sequence)
        df = pd.DataFrame(mutants, columns=['mutant'])

    # Calculate scores using the new function
    scores = esm1v_score(args.fasta_file, mutants)
    df['esm1v_score'] = scores

    # Save results
    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
    else:
        file_name = f"{args.fasta_file.split('/')[-1].split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)

if __name__ == "__main__":
    main()