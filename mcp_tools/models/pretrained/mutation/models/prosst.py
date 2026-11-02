import sys
import os
from pathlib import Path

_REPO_ROOT = next((p for p in Path(__file__).absolute().parents if (p / "src").is_dir()), Path(__file__).absolute().parent)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
import argparse
import torch
import datetime
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from mcp_tools.models.pretrained.prosst.structure.get_sst_seq import SSTPredictor
from mcp_tools.models.pretrained.mutation.utils import generate_mutations_from_sequence
from mcp_tools.models.pretrained.mutation.models.esm.inverse_folding.util import extract_seq_from_pdb
from typing import List
from tqdm import tqdm


def prosst_score(pdb_file: str, mutants: List[str], model_name='AI4Protein/ProSST-2048', structure_weights=None, structure_vocab_size=2048) -> List[float]:
    """
    Calculate ProSST scores for a list of mutations.
    
    Args:
        pdb_file: Path to the PDB file
        mutants: List of mutation strings (e.g., ["A1B", "C2D"])
        
    Returns:
        List of scores corresponding to the input mutations
    """
    if structure_vocab_size not in (20, 128, 512, 1024, 2048, 4096):
        raise ValueError("Unsupported ProSST structure vocabulary")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ProSST model and tokenizer
    prosst_model = AutoModelForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    prosst_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer_weights = Path(structure_weights) if structure_weights else None
    if getattr(prosst_model.config, 'ss_vocab_size', 0) != structure_vocab_size + 3:
        raise ValueError('ProSST checkpoint does not match the selected structure vocabulary')
    predictor = SSTPredictor(structure_vocab_size=structure_vocab_size, num_processes=0, num_threads=1,
        model_path=str(tokenizer_weights / 'AE.pt') if tokenizer_weights else None,
        cluster_dir=str(tokenizer_weights) if tokenizer_weights else None)

    # Extract structure sequence from PDB
    sst_results = predictor.predict_from_pdb(pdb_file)
    if not sst_results or f"{structure_vocab_size}_sst_seq" not in sst_results[0]:
        err = sst_results[0].get("error") if sst_results else None
        detail = f" ({err})" if err else ""
        raise RuntimeError(
            f"ProSST structure tokenization failed for {pdb_file}{detail}. "
            "Check that the PDB has backbone atoms (N/CA/C) and is a valid protein structure."
        )
    structure_sequence = sst_results[0][f"{structure_vocab_size}_sst_seq"]
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
    parser.add_argument('--model_name', default='AI4Protein/ProSST-2048')
    parser.add_argument('--structure_weights', default=None)
    parser.add_argument('--structure_vocab_size', type=int, choices=(20,128,512,1024,2048,4096), default=2048)
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
    scores = prosst_score(args.pdb_file, mutants, args.model_name, args.structure_weights, args.structure_vocab_size)
    df['prosst_score'] = scores
    df = df.sort_values(by='prosst_score', ascending=False)
    
    # Save results
    if args.output_csv is not None:
        output_dir = os.path.dirname(args.output_csv)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
    else:
        file_name = f"{args.pdb_file.split('/')[-1].split('.')[0]}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)

if __name__ == "__main__":
    main()
