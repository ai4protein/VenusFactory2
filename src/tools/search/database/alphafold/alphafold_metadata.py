"""
AlphaFold metadata: query (return JSON text) and download (save to file).
"""
import json
import os
import argparse
import requests

BASE_URL = "https://alphafold.ebi.ac.uk/api/prediction"


def query_alphafold_metadata(uniprot_id: str) -> str:
    """Query AlphaFold prediction metadata by UniProt ID. Returns JSON text. No file save."""
    url = f"{BASE_URL}/{uniprot_id}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            return json.dumps({"success": False, "uniprot_id": uniprot_id, "error": "No AlphaFold prediction found"}, indent=2)
        response.raise_for_status()
        data = response.json()
        return json.dumps(data, indent=2, ensure_ascii=False)
    except requests.RequestException as e:
        return json.dumps({"success": False, "uniprot_id": uniprot_id, "error": str(e)}, indent=2)
    except (ValueError, KeyError) as e:
        return json.dumps({"success": False, "uniprot_id": uniprot_id, "error": str(e)}, indent=2)


def download_alphafold_metadata(uniprot_id: str, out_dir: str) -> str:
    """Download AlphaFold metadata by UniProt ID. Saves JSON file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uniprot_id}.json")
    text = query_alphafold_metadata(uniprot_id)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False and "error" in data:
            return f"{uniprot_id} failed: {data.get('error', 'unknown')}"
    except json.JSONDecodeError:
        return f"{uniprot_id} failed: invalid response"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"{uniprot_id} metadata downloaded"


if __name__ == "__main__":
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--uniprot_id", type=str, help="Single UniProt ID")
    parser.add_argument("-f", "--uniprot_id_file", type=str, help="Input file containing a list of UniProt IDs")
    parser.add_argument("-o", "--out_dir", type=str, help="Output directory to save JSON files")
    parser.add_argument("--output", type=str, help="Single output file path (used with -i only, overrides out_dir)")
    parser.add_argument("-e", "--error_file", type=str, help="File to record failed IDs")
    parser.add_argument("-n", "--num_workers", type=int, default=12)
    args = parser.parse_args()

    if not args.uniprot_id and not args.uniprot_id_file:
        print("Error: Must provide either uniprot_id or uniprot_id_file")
        exit(1)
    if args.output and args.uniprot_id and not args.uniprot_id_file:
        out_dir = os.path.dirname(args.output)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        msg = download_alphafold_metadata(args.uniprot_id, out_dir)
        if "failed" not in msg:
            src = os.path.join(out_dir, f"{args.uniprot_id}.json")
            if os.path.exists(src) and os.path.abspath(src) != os.path.abspath(args.output):
                import shutil
                shutil.move(src, args.output)
        print(msg)
        exit(0)

    if not args.out_dir:
        print("Error: Must provide out_dir (-o) when using -f or batch mode")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    uids = [args.uniprot_id] if args.uniprot_id else [line.strip() for line in open(args.uniprot_id_file) if line.strip()]
    errors, messages = [], []

    if len(uids) == 1:
        msg = download_alphafold_metadata(uids[0], args.out_dir)
        print(msg)
        if "failed" in msg:
            errors.append(uids[0])
            messages.append(msg)
    else:
        fut_to_uid = {}
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            fut_to_uid = {ex.submit(download_alphafold_metadata, uid, args.out_dir): uid for uid in uids}
            for fut in tqdm(as_completed(fut_to_uid), total=len(uids), desc="Downloading"):
                uid = fut_to_uid[fut]
                msg = fut.result()
                if "failed" in msg:
                    errors.append(uid)
                    messages.append(msg)

    if errors and args.error_file:
        d = os.path.dirname(args.error_file)
        if d:
            os.makedirs(d, exist_ok=True)
        pd.DataFrame({"protein": errors, "error": messages}).to_csv(args.error_file, index=False)
