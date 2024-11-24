import argparse
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_fasta(uniprot_id, outdir):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    out_path = os.path.join(outdir, f"{uniprot_id}.fasta")
    if os.path.exists(out_path):
        return uniprot_id, f"{uniprot_id}.fasta already exists, skipping"
    
    if response.status_code != 200:
        return uniprot_id, f"{uniprot_id}.fasta failed, {response.status_code}"

    output_file = os.path.join(outdir, f"{uniprot_id}.fasta")
    with open(output_file, 'w') as file:
        file.write(response.text)
    
    return uniprot_id, f"{uniprot_id}.fasta successfully downloaded"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download FASTA files from UniProt.')
    parser.add_argument('-f', '--file', help='Input file containing UniProt IDs')
    parser.add_argument('-o', '--out_dir', help='Directory to save FASTA files')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='Number of workers to use for downloading')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    uids = open(args.file, 'r').read().splitlines()
    
    error_proteins = []
    error_messages = []
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_fasta = {executor.submit(download_fasta, uid, args.out_dir): uid for uid in uids}

        with tqdm(total=len(uids), desc="Downloading Files") as bar:
            for future in as_completed(future_to_fasta):
                uid, message = future.result()
                bar.set_description(message)
                if "failed" in message:
                    error_proteins.append(uid)
                    error_messages.append(message)
                bar.update(1)
    
    with open(os.path.join(args.out_dir, 'failed.txt'), 'w') as f:
        for protein, message in zip(error_proteins, error_messages):
            f.write(f"{protein} - {message}\n")