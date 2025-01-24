import argparse
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_fasta(uniprot_id, outdir, merge_output=False):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    
    if not merge_output:
        out_path = os.path.join(outdir, f"{uniprot_id}.fasta")
        if os.path.exists(out_path):
            return uniprot_id, f"{uniprot_id}.fasta already exists, skipping", None
    
    if response.status_code != 200:
        return uniprot_id, f"{uniprot_id}.fasta failed, {response.status_code}", None

    if merge_output:
        return uniprot_id, f"{uniprot_id}.fasta successfully downloaded", response.text
    else:
        output_file = os.path.join(outdir, f"{uniprot_id}.fasta")
        with open(output_file, 'w') as file:
            file.write(response.text)
        return uniprot_id, f"{uniprot_id}.fasta successfully downloaded", None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download FASTA files from UniProt.')
    parser.add_argument('-i', '--uniprot_id', help='Single UniProt ID to download')
    parser.add_argument('-f', '--file', help='Input file containing UniProt IDs')
    parser.add_argument('-o', '--out_dir', help='Directory to save FASTA files')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='Number of workers to use for downloading')
    parser.add_argument('-m', '--merge', action='store_true', help='Merge all sequences into a single FASTA file')
    parser.add_argument('-e', '--error_file', help='File to save failed downloads. If not provided, errors will be printed to console')
    args = parser.parse_args()

    if not args.uniprot_id and not args.file:
        print("Error: Must provide either uniprot_id or file")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    error_proteins = []
    error_messages = []
    all_sequences = []
    
    if args.uniprot_id:
        uid, message, sequence = download_fasta(args.uniprot_id, args.out_dir, args.merge)
        print(message)
        if "failed" in message:
            error_proteins.append(uid)
            error_messages.append(message)
        elif args.merge and sequence:
            all_sequences.append(sequence)
    
    elif args.file:
        uids = open(args.file, 'r').read().splitlines()
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_fasta = {executor.submit(download_fasta, uid, args.out_dir, args.merge): uid for uid in uids}

            with tqdm(total=len(uids), desc="Downloading Files") as bar:
                for future in as_completed(future_to_fasta):
                    uid, message, sequence = future.result()
                    bar.set_description(message)
                    if "failed" in message:
                        error_proteins.append(uid)
                        error_messages.append(message)
                    elif args.merge and sequence:
                        all_sequences.append(sequence)
                    bar.update(1)
    
    if args.merge and all_sequences:
        merged_file = os.path.join(args.out_dir, "merged.fasta")
        with open(merged_file, 'w') as f:
            f.write(''.join(all_sequences))
    
    if error_proteins and args.error_file:
        with open(args.error_file, 'w') as f:
            for protein, message in zip(error_proteins, error_messages):
                f.write(f"{protein} - {message}\n")
    elif error_proteins:
        print("Failed downloads:")
        for protein, message in zip(error_proteins, error_messages):
            print(f"{protein} - {message}")