import sys
import os
sys.path.append(os.getcwd())
import argparse
import subprocess
import json
import torch
import datetime
import pandas as pd
from transformers import AutoModelForMaskedLM, AutoTokenizer
from src.data.prosst.structure.get_sst_seq import SSTPredictor
from src.mutation.utils import generate_mutations_from_sequence
from src.mutation.models.esm.inverse_folding.util import extract_seq_from_pdb
from typing import List, Dict
from tqdm import tqdm
from requests import get
from time import sleep


def process_pdb(pdb_file, output_dir):
    """
    Submit PDB to Foldseek API and get structure alignments.
    Returns the path to the generated FASTA file.
    """
    file_name = pdb_file.split('.')[0].split('/')[-1]
    fasta_path = f'{output_dir}/{file_name}.fasta'
    
    if os.path.exists(fasta_path):
        print(f'>>> {file_name} already exists')
        return fasta_path
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Submit a new job and get the ticket
    repeat = True
    try_times = 0
    while repeat:
        result = subprocess.run(
            [
                "curl", "-X", "POST", "-F", f"q=@{pdb_file}", 
                "-F", "mode=3diaa", "-F", "database[]=afdb50", 
                "-F", "database[]=afdb-proteome", "-F", "database[]=cath50", 
                "-F", "database[]=mgnify_esm30", "-F", "database[]=pdb100", 
                "-F", "database[]=gmgcl_id", "-F", "database[]=afdb-swissprot", 
                "-F", "database[]=bfvd", "-F", "database[]=bfmd", 
                "https://search.foldseek.com/api/ticket"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            result = result.stdout
            ticket = json.loads(result)
            repeat = ticket['status'] != 'COMPLETE'
        except:
            sleep(1)
            try_times += 1
            print('>>> Try again for the ' + str(try_times) + ' time')
            if try_times > 10:
                print(f'>>> Failed to submit {file_name} after {try_times} attempts')
                return None
            continue
    
    print('>>> Ticket:', ticket)
    result = get('https://search.foldseek.com/api/result/' + ticket['id'] + '/0').json()
    structure_aln_dict = result
    results = structure_aln_dict['results']
    query_seq = structure_aln_dict['queries'][0]['sequence']
    
    alignment_dict = {}
    for result_db in results:
        if len(result_db['alignments']) == 0:
            continue
        for target_info in result_db['alignments'][0]:
            name = f"{target_info['target']}/prob_{target_info['prob']}/eval_{target_info['eval']}/score_{target_info['score']}/{target_info['qStartPos']}-{target_info['qEndPos']}"
            qaln = target_info['qAln']
            dbaln = target_info['dbAln']
            try:
                # Get the index list of '-' in qaln
                qaln = list(qaln)
                miss_index = [i for i in range(len(qaln)) if qaln[i] == '-']
                # Remove the residues in dbaln according to the missing index list in qaln
                dbaln = list(dbaln)
                dbaln = ''.join([dbaln[i] for i in range(len(dbaln)) if i not in miss_index])
            except Exception as e:
                print(e)
                print(name)
            # Fill '-' to the left and right of dbaln to make it the same length as query_seq
            left = target_info['qStartPos'] - 1
            right = len(query_seq) - target_info['qEndPos']
            dbaln = '-' * left + dbaln + '-' * right
            alignment_dict[name] = dbaln
    
    if alignment_dict == {}:
        print(f'>>> {file_name} has no alignment')
        with open(fasta_path, 'w') as f:
            f.write('\n')
        return None
    
    seqs = []
    with open(fasta_path, 'w') as f:
        for key, value in alignment_dict.items():
            if value not in seqs:
                seqs.append(value)
            else:
                continue
            f.write(f'>{key}\n{value}\n')
    
    print(f'>>> {file_name} done, saved {len(seqs)} unique alignments')
    return fasta_path


def read_multi_fasta(fasta_file: str) -> Dict[str, str]:
    """
    Read a multi-FASTA file and return a dictionary of sequences.
    """
    alignment_dict = {}
    if not os.path.exists(fasta_file):
        return alignment_dict
    
    with open(fasta_file, 'r') as f:
        current_name = None
        current_seq = []
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name is not None:
                    alignment_dict[current_name] = ''.join(current_seq)
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line)
        # Don't forget the last sequence
        if current_name is not None:
            alignment_dict[current_name] = ''.join(current_seq)
    
    return alignment_dict


def count_matrix_from_structure_alignment(tokenizer, alignment_dict, device):
    """
    Convert structure alignment to count matrix and then to log probability distribution.
    Returns a tensor of shape [seq_len, vocab_size] with log probabilities.
    """
    alignment_seqs = list(alignment_dict.values())
    print(f">>> Start tokenizing {len(alignment_seqs)} structure alignment sequences")
    
    if len(alignment_seqs) == 0:
        return None
    
    tokenized_results = tokenizer(alignment_seqs, return_tensors="pt", padding=True)
    alignment_ids = tokenized_results["input_ids"][:, 1:-1]  # Remove [CLS] and [SEP] tokens
    
    # Count distribution of each column, [seq_len, vocab_size]
    count_matrix = torch.zeros(alignment_ids.size(1), tokenizer.vocab_size)
    print(f">>> Counting amino acid distribution at each position")
    for i in tqdm(range(alignment_ids.size(1))):
        count_matrix[i] = torch.bincount(alignment_ids[:, i], minlength=tokenizer.vocab_size)
    
    # Convert counts to probabilities
    count_matrix = count_matrix / count_matrix.sum(dim=1, keepdim=True)
    count_matrix = count_matrix.to(device)
    
    # Apply log_softmax for numerical stability
    count_matrix = torch.log_softmax(count_matrix, dim=-1)
    
    return count_matrix


def venusrem_score(pdb_file: str, mutants: List[str], alpha: float = 0.8) -> List[float]:
    """
    Calculate VenusREM scores for a list of mutations.
    
    Args:
        pdb_file: Path to the PDB file
        mutants: List of mutation strings (e.g., ["A1B", "C2D"])
        alpha: Weight for structure alignment information (default: 0.8)
        
    Returns:
        List of scores corresponding to the input mutations
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ProSST model and tokenizer
    print(">>> Loading ProSST model and tokenizer...")
    prosst_model = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True).to(device)
    prosst_tokenizer = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True)
    predictor = SSTPredictor(structure_vocab_size=2048)

    # Extract structure sequence from PDB
    print(">>> Extracting structure sequence from PDB...")
    structure_sequence = predictor.predict_from_pdb(pdb_file)[0]['2048_sst_seq']
    structure_sequence_offset = [i + 3 for i in structure_sequence]

    # Extract residue sequence from PDB
    print(">>> Extracting residue sequence from PDB...")
    residue_sequence = extract_seq_from_pdb(pdb_file)

    # Tokenize sequences
    tokenized_res = prosst_tokenizer([residue_sequence], return_tensors='pt')
    input_ids = tokenized_res['input_ids'].to(device)
    attention_mask = tokenized_res['attention_mask'].to(device)
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0).to(device)

    # Compute ProSST logits
    print(">>> Computing ProSST logits...")
    with torch.no_grad():
        outputs = prosst_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ss_input_ids=structure_input_ids
        )
    logits = torch.log_softmax(outputs.logits[:, 1:-1], dim=-1).squeeze()

    # Process Foldseek structure alignment
    print(">>> Submitting to Foldseek for structure alignment...")
    output_dir = os.path.join(os.path.dirname(pdb_file), 'foldseek_alignments')
    fasta_path = process_pdb(pdb_file, output_dir)
    
    # If Foldseek alignment is available, incorporate it
    if fasta_path is not None:
        alignment_dict = read_multi_fasta(fasta_path)
        
        if len(alignment_dict) > 0:
            print(f">>> Processing {len(alignment_dict)} structure alignments...")
            count_matrix = count_matrix_from_structure_alignment(prosst_tokenizer, alignment_dict, device)
            
            if count_matrix is not None:
                # Ensure dimensions match
                if count_matrix.size(0) == logits.size(0):
                    print(f">>> Combining ProSST logits with structure alignment (alpha={alpha})...")
                    logits = (1 - alpha) * logits + alpha * count_matrix
                else:
                    print(f">>> Warning: Dimension mismatch between logits ({logits.size(0)}) and count_matrix ({count_matrix.size(0)})")
                    print(f">>> Using ProSST logits only")
        else:
            print(">>> No valid alignments found, using ProSST logits only")
    else:
        print(">>> No Foldseek alignment available, using ProSST logits only")

    # Get vocabulary for scoring
    vocab = prosst_tokenizer.get_vocab()
    
    # Calculate scores for each mutation
    print(">>> Calculating mutation scores...")
    pred_scores = []
    for mutant in tqdm(mutants):
        mutant_score = 0
        sep = ":" if ":" in mutant else ";"
        mutations = mutant.split(sep)
        
        for sub_mutant in mutations:
            wt, idx, mt = sub_mutant[0], int(sub_mutant[1:-1]) - 1, sub_mutant[-1]
            # Calculate log probability difference: log(P(mutant)) - log(P(wildtype))
            pred = logits[idx, vocab[mt]] - logits[idx, vocab[wt]]
            mutant_score += pred.item()
        
        # Average score for multiple mutations
        pred_scores.append(mutant_score / len(mutations))

    return pred_scores


def main():
    parser = argparse.ArgumentParser(description='VenusREM: ProSST with Foldseek structure alignment')
    parser.add_argument('--pdb_file', type=str, required=True, help='Path to the PDB file')
    parser.add_argument('--mutations_csv', type=str, default=None, help='Path to the mutations CSV file')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to the output CSV file')
    parser.add_argument('--alpha', type=float, default=0.8, help='Weight for structure alignment (default: 0.8)')
    args = parser.parse_args()

    # Extract residue sequence from PDB
    residue_sequence = extract_seq_from_pdb(args.pdb_file)

    # Handle mutations
    if args.mutations_csv is not None:
        df = pd.read_csv(args.mutations_csv)
        mutants = df['mutant'].tolist()
    else:
        print(">>> Generating all single mutations...")
        mutants = generate_mutations_from_sequence(residue_sequence)
        df = pd.DataFrame(mutants, columns=['mutant'])

    # Calculate scores using VenusREM
    scores = venusrem_score(args.pdb_file, mutants, alpha=args.alpha)
    df['venusrem_score'] = scores
    df = df.sort_values(by='venusrem_score', ascending=False)
    
    # Save results
    if args.output_csv is not None:
        output_dir = os.path.dirname(args.output_csv)
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f">>> Results saved to {args.output_csv}")
    else:
        file_name = f"{args.pdb_file.split('/')[-1].split('.')[0]}_venusrem_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(file_name, index=False)
        print(f">>> Results saved to {file_name}")


if __name__ == "__main__":
    main()