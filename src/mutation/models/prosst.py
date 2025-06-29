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
from src.data.prosst.structure.utils.data_utils import extract_seq_from_pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prosst')
    parser.add_argument('--pdb_file', type=str, required=True, help='Path to the pdb file')
    parser.add_argument('--mutations_csv', type=str, default=None, help='Path to the mutations CSV file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to the output CSV file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prosst_model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True).to(device)
    prosst_tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
    predictor = SSTPredictor(structure_vocab_size=2048)

    structure_sequence = predictor.predict_from_pdb(args.pdb_file)[0]['2048_sst_seq']
    structure_sequence_offset = [i + 3 for i in structure_sequence]

    residue_sequence = extract_seq_from_pdb(args.pdb_file)

    tokenized_res = prosst_tokenizer([residue_sequence], return_tensors='pt')
    input_ids = tokenized_res['input_ids'].to(device)
    attention_mask = tokenized_res['attention_mask'].to(device)
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)

    with torch.no_grad():
        outputs = prosst_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=structure_input_ids
        )
    logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()

    if args.mutations_csv is not None:
        df = pd.read_csv(args.mutations_csv)
        mutants = df['mutant'].tolist()
    else:
        mutants = generate_mutations_from_sequence(residue_sequence)
        df = pd.DataFrame(mutants, columns=['mutant'])

    vocab = prosst_tokenizer.get_vocab()
    pred_scores = []
    for mutant in mutants:
        mutant_score = 0
        for sub_mutant in mutant.split(":"):
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            pred = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
            mutant_score += pred.item()
        pred_scores.append(mutant_score)

    df['prosst_score'] = pred_scores

    if args.output_csv is not None:
        df.to_csv(args.output_csv, index=False)
    else:
        file_name = f"{args.pdb_file.split('/')[-1].split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)