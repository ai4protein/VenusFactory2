"""
RCSB PDB metadata: query (return JSON text) and download (save to file).
"""
import json
import os
import argparse
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

RCSB_ENTRY_BASE = "https://data.rcsb.org/rest/v1/core/entry"


def query_rcsb_entry(pdb_id: str) -> str:
    """Query RCSB entry metadata by PDB ID. Returns JSON text or error JSON string. No file save."""
    url = f"{RCSB_ENTRY_BASE}/{pdb_id.upper()}"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return json.dumps(data, indent=2) if data else json.dumps({"error": "Empty response"})
        return json.dumps({"error": f"HTTP {response.status_code}", "pdb_id": pdb_id})
    except requests.RequestException as e:
        return json.dumps({"error": str(e), "pdb_id": pdb_id})
    except (ValueError, TypeError):
        return json.dumps({"error": "Invalid JSON", "pdb_id": pdb_id})


def get_metadata_from_rcsb(pdb_id: str):
    """Legacy: return (data, message) or (None, error_message)."""
    text = query_rcsb_entry(pdb_id)
    try:
        data = json.loads(text)
        if data and "error" not in data:
            return data, f"{pdb_id} successfully downloaded"
    except json.JSONDecodeError:
        pass
    return None, f"{pdb_id} failed to download (empty or error)"


def download_rcsb_entry(pdb_id: str, out_dir: str) -> str:
    """Download RCSB entry metadata by PDB ID. Saves JSON file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pdb_id}.json")
    if os.path.exists(out_path):
        return f"Skipping {pdb_id}, already exists"
    text = query_rcsb_entry(pdb_id)
    try:
        data = json.loads(text)
        if data and "error" not in data:
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            return f"{pdb_id} successfully downloaded"
    except json.JSONDecodeError:
        pass
    return f"{pdb_id} failed to download (empty or error)"


def download_single_pdb(pdb_id: str, out_dir: str) -> str:
    """Alias for download_rcsb_entry (backward compatibility)."""
    return download_rcsb_entry(pdb_id, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_id_file", type=str, default=None)
    parser.add_argument("--pdb_id", type=str, default=None)
    parser.add_argument("--error_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=12)
    args = parser.parse_args()

    if not args.pdb_id and not args.pdb_id_file:
        print("Error: Must provide either pdb_id or pdb_id_file")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    downloaded = [p.replace(".json", "") for p in os.listdir(args.out_dir) if p.endswith(".json")]
    errors, messages = [], []

    if args.pdb_id_file:
        with open(args.pdb_id_file) as f:
            pdbs = [line.strip() for line in f if line.strip()]

        def _dl(pdb_id, downloaded, out_dir):
            if pdb_id in downloaded:
                return pdb_id, f"{pdb_id} already exists, skipping"
            return pdb_id, download_rcsb_entry(pdb_id, out_dir)

        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(_dl, pdb, downloaded, args.out_dir): pdb for pdb in pdbs}
            with tqdm(total=len(pdbs), desc="Downloading RCSB Entry") as bar:
                for fut in as_completed(futures):
                    pdb_id, msg = fut.result()
                    bar.set_description(msg)
                    if "failed" in msg:
                        errors.append(pdb_id)
                        messages.append(msg)
                    bar.update(1)
    else:
        msg = download_rcsb_entry(args.pdb_id, args.out_dir)
        print(msg)
        if "failed" in msg:
            errors.append(args.pdb_id)
            messages.append(msg)

    if errors and args.error_file:
        d = os.path.dirname(args.error_file)
        os.makedirs(d, exist_ok=True)
        pd.DataFrame({"protein": errors, "error": messages}).to_csv(args.error_file, index=False)
