import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
import datetime
import pandas as pd
from tqdm import tqdm
from vplm import TransformerForMaskedLM, TransformerConfig
from vplm import VPLMTokenizer
from src.mutation.utils import generate_mutations_from_sequence
from typing import List

amino_acids = "LAGVSERTIDPKQNFYMHWC"

def venusplm_score(fasta_file: str, mutants: List[str],
                    model_name: str = "AI4Protein/VenusPLM-300M") -> List[float]:
    """
    Calculate VenusPLM scores for a list of mutations

    Args:
        fasta_file: Path to the Fasta file
        mutants: List of mutation strings (e.g., ["A1B", "C2D"])
        model_name: VenusPLM model name

    Returns:
        List of scores corresponding to the input mutations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load VenusPLM model and tokenizer
    venusplm_tokenizer = VPLMTokenizer.from_pretrained("AI4Protein/VenusPLM-300M")
    venusplm_model = TransformerForMaskedLM.from_pretrained("AI4Protein/VenusPLM-300M").to(device)

    # Load sequence from FASTA file
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
        sequence = ''.join(line.strip() for line in lines[1:])

    # Tokenize sequence
    tokenized_res = venusplm_tokenizer([sequence], return_tensors="pt").to(device)
    vocab = venusplm_tokenizer.get_vocab()

    # Compute logits
    with torch.no_grad():
        outputs = venusplm_model(
            input_ids=tokenized_res['input_ids'],
            attention_mask = tokenized_res["attention_mask"],
            output_hidden_states = False
        )
        logits = outputs.logits.log_softmax(dim=-1).squeeze()[1:-1]
    
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
        pred_scores.append(mutant_score /  len(mutant.split(sep)))
    
    return pred_scores

def main():
    parser = argparse.ArgumentParser(description='VenusPLM')
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
    scores = venusplm_score(args.fasta_file, mutants)
    df['venusplm_score'] = scores
    df = df.sort_values(by='venusplm_score', ascending=False)
    
    # Save results
    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
    else:
        file_name = f"{args.fasta_file.split('/')[-1].split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)

if __name__ == "__main__":
    main()