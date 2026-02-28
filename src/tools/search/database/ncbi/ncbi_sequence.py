"""NCBI sequence: query (return FASTA text) and download (save to file)."""
import argparse
import json
import os
import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def query_ncbi_seq(ncbi_id: str, db: str = "protein") -> str:
    """Query NCBI sequence by accession. Returns FASTA text or error JSON string. No file save."""
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={db}&id={ncbi_id}&rettype=fasta&retmode=text"
    try:
        response = requests.get(url, timeout=30)
        time.sleep(0.4)
        if response.status_code != 200:
            return json.dumps({"success": False, "ncbi_id": ncbi_id, "error": f"HTTP {response.status_code}"})
        if not response.text or "Error" in response.text:
            return json.dumps({"success": False, "ncbi_id": ncbi_id, "error": "ID not found or invalid"})
        return response.text.strip()
    except requests.RequestException as e:
        return json.dumps({"success": False, "ncbi_id": ncbi_id, "error": str(e)})


def download_ncbi_seq(ncbi_id: str, out_dir: str, db: str = "protein") -> str:
    """Download NCBI sequence by accession. Saves FASTA file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ncbi_id}.fasta")
    if os.path.exists(out_path):
        return f"Skipping {ncbi_id}, already exists"
    text = query_ncbi_seq(ncbi_id, db)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return f"{ncbi_id} failed: {data.get('error', 'unknown')}"
    except json.JSONDecodeError:
        pass
    with open(out_path, "w") as f:
        f.write(text)
    return f"{ncbi_id} successfully downloaded"


def download_ncbi_fasta(ncbi_id, db, outdir, merge_output=False):
    """Legacy API: (ncbi_id, status_message, sequence_or_None)."""
    if not merge_output:
        out_path = os.path.join(outdir, f"{ncbi_id}.fasta")
        if os.path.exists(out_path):
            return ncbi_id, f"{ncbi_id}.fasta already exists, skipping", None
    text = query_ncbi_seq(ncbi_id, db)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return ncbi_id, f"{ncbi_id}.fasta failed, {data.get('error', 'unknown')}", None
    except json.JSONDecodeError:
        pass
    if merge_output:
        return ncbi_id, f"{ncbi_id}.fasta successfully downloaded", text
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"{ncbi_id}.fasta")
    with open(out_path, "w") as f:
        f.write(text)
    return ncbi_id, f"{ncbi_id}.fasta successfully downloaded", None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download FASTA files from the NCBI database.")
    parser.add_argument("-i", "--id", dest="ncbi_id", help="Single NCBI ID to download")
    parser.add_argument("-f", "--file", help="Input file containing a list of NCBI IDs (one per line)")
    parser.add_argument("-o", "--out_dir", required=True, help="Directory to save FASTA files")
    parser.add_argument("-d", "--db", default="protein", help="NCBI database to use (e.g., 'protein', 'nuccore')")
    parser.add_argument("-n", "--num_workers", type=int, default=10, help="Number of parallel workers for downloading")
    parser.add_argument("-m", "--merge", action="store_true", help="Merge all sequences into a single FASTA file named merged.fasta")
    parser.add_argument("-e", "--error_file", help="File to save failed download IDs and messages. If not provided, errors are printed to the console.")
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

    if args.ncbi_id:
        uid, message, sequence = download_ncbi_fasta(args.ncbi_id, args.db, args.out_dir, args.merge)
        print(message)
        if "failed" in message:
            error_entries.append(uid)
            error_messages.append(message)
        elif args.merge and sequence:
            all_sequences.append(sequence)

    elif args.file:
        with open(args.file) as f:
            ids = [line.strip() for line in f if line.strip()]

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_id = {executor.submit(download_ncbi_fasta, ncbi_id, args.db, args.out_dir, args.merge): ncbi_id for ncbi_id in ids}

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

    if args.merge and all_sequences:
        merged_file_path = os.path.join(args.out_dir, "merged.fasta")
        with open(merged_file_path, "w", encoding="utf-8") as f:
            f.write("".join(all_sequences))
        print(f"\nAll sequences successfully merged into {merged_file_path}")

    if error_entries:
        print(f"\nEncountered {len(error_entries)} errors during download.")
        if args.error_file:
            with open(args.error_file, "w", encoding="utf-8") as f:
                for entry, message in zip(error_entries, error_messages):
                    f.write(f"{entry}\t{message}\n")
            print(f"Details for failed downloads saved to {args.error_file}")
        else:
            print("\n--- Failed Downloads ---")
            for entry, message in zip(error_entries, error_messages):
                print(f"{entry} - {message}")
