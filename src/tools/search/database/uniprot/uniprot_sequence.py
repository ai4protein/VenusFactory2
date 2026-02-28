"""
UniProt sequence: query (return FASTA text) and download (save to file).
"""
import json
import os
import argparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def query_uniprot_seq(uniprot_id: str) -> str:
    """Query UniProt sequence by ID. Returns FASTA text or error JSON string. No file save."""
    url = f"https://www.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            return json.dumps({"success": False, "uniprot_id": uniprot_id, "error": "Not found"})
        response.raise_for_status()
        return response.text.strip() if response.text else json.dumps({"success": False, "uniprot_id": uniprot_id, "error": "Empty"})
    except requests.RequestException as e:
        return json.dumps({"success": False, "uniprot_id": uniprot_id, "error": str(e)})


def download_uniprot_seq(uniprot_id: str, out_dir: str) -> str:
    """Download UniProt sequence by ID. Saves FASTA file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uniprot_id}.fasta")
    if os.path.exists(out_path):
        return f"Skipping {uniprot_id}, already exists"
    text = query_uniprot_seq(uniprot_id)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return f"{uniprot_id} failed: {data.get('error', 'unknown')}"
    except json.JSONDecodeError:
        pass
    with open(out_path, "w") as f:
        f.write(text)
    return f"{uniprot_id} successfully downloaded"


def download_uniprot_sequence(uniprot_id: str, outdir: str = None, merge_output: bool = False) -> dict:
    """Legacy API: fetch UniProt sequence. Returns dict with success, uniprot_id, path/sequence/message."""
    text = query_uniprot_seq(uniprot_id)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return {"success": False, "uniprot_id": uniprot_id, "error_message": data.get("error", "unknown")}
    except json.JSONDecodeError:
        pass
    seq = "".join(line.strip() for line in text.split("\n") if line and not line.startswith(">"))
    if outdir is None:
        return {"success": True, "uniprot_id": uniprot_id, "sequence": seq}
    os.makedirs(outdir, exist_ok=True)
    if not merge_output:
        out_path = os.path.join(outdir, f"{uniprot_id}.fasta")
        if os.path.exists(out_path):
            return {"success": True, "uniprot_id": uniprot_id, "path": out_path, "message": f"{uniprot_id}.fasta already exists, skipping"}
        with open(out_path, "w") as f:
            f.write(text)
        return {"success": True, "uniprot_id": uniprot_id, "path": out_path, "message": f"{uniprot_id}.fasta successfully downloaded"}
    return {"success": True, "uniprot_id": uniprot_id, "fasta_text": text, "message": f"{uniprot_id}.fasta successfully downloaded"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--uniprot_id")
    parser.add_argument("-f", "--file")
    parser.add_argument("-o", "--out_dir", required=True)
    parser.add_argument("-n", "--num_workers", type=int, default=12)
    parser.add_argument("-m", "--merge", action="store_true")
    parser.add_argument("-e", "--error_file")
    args = parser.parse_args()

    if not args.uniprot_id and not args.file:
        print("Error: Must provide either uniprot_id or file")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    errors, messages, all_seqs = [], [], []
    uids = [args.uniprot_id] if args.uniprot_id else [line.strip() for line in open(args.file) if line.strip()]

    if args.uniprot_id:
        r = download_uniprot_sequence(args.uniprot_id, args.out_dir, args.merge)
        print(r.get("message", r.get("error_message", "")))
        if not r["success"]:
            errors.append(r["uniprot_id"])
            messages.append(r.get("error_message", ""))
        elif args.merge and r.get("fasta_text"):
            all_seqs.append(r["fasta_text"])
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(download_uniprot_sequence, uid, args.out_dir, args.merge): uid for uid in uids}
            for fut in tqdm(as_completed(futures), total=len(uids), desc="Downloading"):
                r = fut.result()
                if not r["success"]:
                    errors.append(r["uniprot_id"])
                    messages.append(r.get("error_message", ""))
                elif args.merge and r.get("fasta_text"):
                    all_seqs.append(r["fasta_text"])

    if args.merge and all_seqs:
        merged = os.path.join(args.out_dir, "merged.fasta")
        with open(merged, "w") as f:
            f.write("".join(all_seqs))

    if errors and args.error_file:
        with open(args.error_file, "w") as f:
            for p, m in zip(errors, messages):
                f.write(f"{p} - {m}\n")
