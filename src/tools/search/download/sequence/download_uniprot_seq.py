import argparse
import requests
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def download_uniprot_sequence(uniprot_id: str, outdir: str = None, merge_output: bool = False) -> dict:
    """
    Fetch/download protein sequence for a single UniProt ID (uniprotkb API).
    Returns dict:
      - outdir None (fetch only): {success, uniprot_id, sequence} or {success, uniprot_id, error_message}
      - outdir set, not merge: {success, uniprot_id, path, message} or error dict
      - outdir set, merge: {success, uniprot_id, fasta_text, message} or error dict
    """
    url = f"https://www.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        raw = response.text
        lines = raw.strip().split("\n")
        sequence = "".join(line.strip() for line in lines if line and not line.startswith(">"))
    except Exception as e:
        return {"success": False, "uniprot_id": uniprot_id, "error_message": str(e)}

    if outdir is None:
        return {"success": True, "uniprot_id": uniprot_id, "sequence": sequence}

    os.makedirs(outdir, exist_ok=True)
    if not merge_output:
        out_path = os.path.join(outdir, f"{uniprot_id}.fasta")
        if os.path.exists(out_path):
            return {"success": True, "uniprot_id": uniprot_id, "path": out_path, "message": f"{uniprot_id}.fasta already exists, skipping"}
        with open(out_path, "w") as f:
            f.write(raw)
        return {"success": True, "uniprot_id": uniprot_id, "path": out_path, "message": f"{uniprot_id}.fasta successfully downloaded"}
    return {"success": True, "uniprot_id": uniprot_id, "fasta_text": raw, "message": f"{uniprot_id}.fasta successfully downloaded"}

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
    
    if args.merge:
        merged_file = os.path.join(args.out_dir, "merged.fasta")
        if os.path.exists(merged_file):
            print(f"Warning: {merged_file} already exists, skipping merge")
            exit(0)
    
    if args.uniprot_id:
        r = download_uniprot_sequence(args.uniprot_id, args.out_dir, args.merge)
        print(r.get("message", r.get("error_message", "")))
        if not r["success"]:
            error_proteins.append(r["uniprot_id"])
            error_messages.append(r.get("error_message", ""))
        elif args.merge and r.get("fasta_text"):
            all_sequences.append(r["fasta_text"])

    elif args.file:
        uids = open(args.file, 'r').read().splitlines()
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_r = {executor.submit(download_uniprot_sequence, uid, args.out_dir, args.merge): uid for uid in uids}
            with tqdm(total=len(uids), desc="Downloading Files") as bar:
                for future in as_completed(future_to_r):
                    r = future.result()
                    bar.set_description(r.get("message", r.get("error_message", "")))
                    if not r["success"]:
                        error_proteins.append(r["uniprot_id"])
                        error_messages.append(r.get("error_message", ""))
                    elif args.merge and r.get("fasta_text"):
                        all_sequences.append(r["fasta_text"])
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