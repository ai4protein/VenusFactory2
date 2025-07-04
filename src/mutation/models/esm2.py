import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
import datetime
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.mutation.utils import generate_mutations_from_sequence


def main():
    parser = argparse.ArgumentParser(description='ESM2')
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to the fasta file')
    parser.add_argument('--mutations_csv', type=str, default=None, help='Path to the mutations CSV file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to the output CSV file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esm2_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", trust_remote_code=True).to(device)
    esm2_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", trust_remote_code=True)

    with open(args.fasta_file, 'r') as f:
        lines = f.readlines()
        sequence = ''.join(line.strip() for line in lines[1:])

    tokenized_res = esm2_tokenizer([sequence], return_tensors='pt')
    vocab = esm2_tokenizer.get_vocab()
    input_ids = tokenized_res['input_ids'].to(device)
    attention_mask = tokenized_res['attention_mask'].to(device)

    with torch.no_grad():
        outputs = esm2_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()

    if args.mutations_csv is not None:
        df = pd.read_csv(args.mutations_csv)
        mutants = df['mutant'].tolist()
    else:
        mutants = generate_mutations_from_sequence(sequence)
        df = pd.DataFrame(mutants, columns=['mutant'])

    pred_scores = []
    for mutant in mutants:
        mutant_score = 0
        sep = ":" if ":" in mutant else ";"
        for sub_mutant in mutant.split(sep):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            pred = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
            mutant_score += pred.item()
        pred_scores.append(mutant_score / len(mutant.split(sep)))

    df['esm2_score'] = pred_scores

    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
    else:
        file_name = f"{args.fasta_file.split('/')[-1].split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)

if __name__ == "__main__":
    main()