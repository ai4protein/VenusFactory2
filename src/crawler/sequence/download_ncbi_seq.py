import argparse
import requests
import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_ncbi_fasta(ncbi_id, db, outdir, merge_output=False):
    """
    Downloads a FASTA file from the NCBI database for a given ID.

    Args:
        ncbi_id (str): The NCBI accession ID (e.g., 'NP_000517.1').
        db (str): The NCBI database to query (e.g., 'protein', 'nuccore').
        outdir (str): The directory to save the file.
        merge_output (bool): If True, returns the sequence content instead of writing to a file.

    Returns:
        tuple: A tuple containing the ncbi_id, a status message, and the sequence content (if merge_output is True).
    """
    # NCBI E-utilities URL for fetching FASTA files
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={db}&id={ncbi_id}&rettype=fasta&retmode=text"
    
    # If not merging, check if the file already exists
    if not merge_output:
        out_path = os.path.join(outdir, f"{ncbi_id}.fasta")
        if os.path.exists(out_path):
            return ncbi_id, f"{ncbi_id}.fasta already exists, skipping", None

    try:
        response = requests.get(url)
        # NCBI recommends not exceeding 3 requests per second without an API key.
        # A small delay helps to stay within this limit.
        time.sleep(0.4) 

        if response.status_code != 200:
            return ncbi_id, f"{ncbi_id}.fasta failed, HTTP {response.status_code}", None
        
        # Check for empty or error responses from NCBI
        if not response.text or "Error" in response.text:
            return ncbi_id, f"{ncbi_id}.fasta failed, ID not found or invalid", None

        if merge_output:
            return ncbi_id, f"{ncbi_id}.fasta successfully downloaded", response.text
        else:
            output_file = os.path.join(outdir, f"{ncbi_id}.fasta")
            with open(output_file, 'w', encoding='utf-8') as file:
                file.write(response.text)
            return ncbi_id, f"{ncbi_id}.fasta successfully downloaded", None

    except requests.exceptions.RequestException as e:
        return ncbi_id, f"{ncbi_id}.fasta failed, RequestException: {e}", None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download FASTA files from the NCBI database.')
    parser.add_argument('-i', '--id', dest='ncbi_id', help='Single NCBI ID to download')
    parser.add_argument('-f', '--file', help='Input file containing a list of NCBI IDs (one per line)')
    parser.add_argument('-o', '--out_dir', required=True, help='Directory to save FASTA files')
    parser.add_argument('-d', '--db', default='protein', help="NCBI database to use (e.g., 'protein', 'nuccore')")
    parser.add_argument('-n', '--num_workers', type=int, default=10, help='Number of parallel workers for downloading')
    parser.add_argument('-m', '--merge', action='store_true', help='Merge all sequences into a single FASTA file named merged.fasta')
    parser.add_argument('-e', '--error_file', help='File to save failed download IDs and messages. If not provided, errors are printed to the console.')
    args = parser.parse_args()

    if not args.ncbi_id and not args.file:
        parser.error("Error: Must provide either a single NCBI ID with --id or a file with --file")

    os.makedirs(args.out_dir, exist_ok=True)
    error_entries = []
    error_messages = []
    all_sequences = []
    
    if args.merge:
        merged_file = os.path.join(args.out_dir, "merged.fasta")
        if os.path.exists(merged_file):
            print(f"Warning: Merged file {merged_file} already exists. To prevent overwriting, the script will exit. Please remove the existing file or change the output directory.")
            exit(0)
    
    # Handle a single ID download
    if args.ncbi_id:
        uid, message, sequence = download_ncbi_fasta(args.ncbi_id, args.db, args.out_dir, args.merge)
        print(message)
        if "failed" in message:
            error_entries.append(uid)
            error_messages.append(message)
        elif args.merge and sequence:
            all_sequences.append(sequence)
    
    # Handle downloading from a file of IDs
    elif args.file:
        with open(args.file, 'r') as f:
            ids = [line.strip() for line in f if line.strip()]
        
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Create a future for each download task
            future_to_id = {executor.submit(download_ncbi_fasta, ncbi_id, args.db, args.out_dir, args.merge): ncbi_id for ncbi_id in ids}

            # Process results as they complete, with a progress bar
            with tqdm(total=len(ids), desc="Downloading NCBI Files") as bar:
                for future in as_completed(future_to_id):
                    uid, message, sequence = future.result()
                    bar.set_description(f"Processing {uid}")
                    if "failed" in message:
                        error_entries.append(uid)
                        error_messages.append(message)
                    elif args.merge and sequence:
                        all_sequences.append(sequence)
                    bar.update(1)
    
    # If merging is enabled, write all collected sequences to the merged file
    if args.merge and all_sequences:
        merged_file_path = os.path.join(args.out_dir, "merged.fasta")
        with open(merged_file_path, 'w', encoding='utf-8') as f:
            f.write(''.join(all_sequences))
        print(f"\nAll sequences successfully merged into {merged_file_path}")
    
    # Handle logging of failed downloads
    if error_entries:
        print(f"\nEncountered {len(error_entries)} errors during download.")
        if args.error_file:
            with open(args.error_file, 'w', encoding='utf-8') as f:
                for entry, message in zip(error_entries, error_messages):
                    f.write(f"{entry}\t{message}\n")
            print(f"Details for failed downloads saved to {args.error_file}")
        else:
            print("\n--- Failed Downloads ---")
            for entry, message in zip(error_entries, error_messages):
                print(f"{entry} - {message}")