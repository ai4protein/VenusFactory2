import sys
import os
sys.path.append(os.getcwd())
import argparse
import torch
import datetime
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.mutation.utils import generate_mutations_from_sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ESM1V')
    parser.add_argument('--fasta_file', type=str, required=True, help='Path to the fasta file')
    parser.add_argument('--mutations_csv', type=str, default=None, help='Path to the mutations CSV file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to the output CSV file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esm1v_model = AutoModelForMaskedLM.from_pretrained("facebook/esm1v_t33_650M_UR90S_1", trust_remote_code=True).to(device)
    esm1v_tokenizer = AutoTokenizer.from_pretrained("facebook/esm1v_t33_650M_UR90S_1", trust_remote_code=True)

    with open(args.fasta_file, 'r') as f:
        lines = f.readlines()
        sequence = ''.join(line.strip() for line in lines[1:])

    tokenized_res = esm1v_tokenizer([sequence], return_tensors='pt')
    vocab = esm1v_tokenizer.get_vocab()
    input_ids = tokenized_res['input_ids'].to(device)
    attention_mask = tokenized_res['attention_mask'].to(device)

    with torch.no_grad():
        outputs = esm1v_model(input_ids=input_ids, attention_mask=attention_mask)
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
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            pred = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
            mutant_score += pred.item()
        pred_scores.append(mutant_score)

    df['esm1v_score'] = pred_scores

    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
    else:
        file_name = f"{args.fasta_file.split('/')[-1].split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)