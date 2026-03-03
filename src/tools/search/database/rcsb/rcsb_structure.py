"""
RCSB PDB structure: query (return structure text) and download (save to file).
"""
import gzip
import json
import os
import argparse
import requests
import shutil
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

RCSB_FILES_BASE = "https://files.rcsb.org/download"
SUFFIX_MAP = {"pdb": "pdb.gz", "cif": "cif.gz", "pdb1": "pdb1.gz", "xml": "xml.gz", "sf": "-sf.cif.gz", "mr": "mr.gz", "mrstr": "_mr.str.gz"}


def query_rcsb_structure(pdb_id: str, file_type: str = "pdb") -> str:
    """Query RCSB structure by PDB ID. Returns structure text or error JSON string. No file save."""
    suffix = SUFFIX_MAP.get(file_type, "pdb.gz")
    url = f"{RCSB_FILES_BASE}/{pdb_id.upper()}.{suffix}"
    try:
        response = requests.get(url, timeout=60, stream=True)
        if response.status_code != 200:
            return json.dumps({"error": f"HTTP {response.status_code}", "pdb_id": pdb_id})
        raw = response.content
        if not raw:
            return json.dumps({"error": "Empty response", "pdb_id": pdb_id})
        try:
            return gzip.decompress(raw).decode("utf-8", errors="replace")
        except OSError:
            return raw.decode("utf-8", errors="replace")
    except requests.RequestException as e:
        return json.dumps({"error": str(e), "pdb_id": pdb_id})


def download_rcsb_structure(pdb_id: str, out_dir: str, file_type: str = "pdb", unzip: bool = True):
    """Download RCSB structure by PDB ID. Saves file. Returns path or None on failure."""
    os.makedirs(out_dir, exist_ok=True)
    pdb_id = pdb_id.upper()
    suffix = SUFFIX_MAP.get(file_type, "pdb.gz")
    out_name = f"{pdb_id}.{suffix}"
    out_path = os.path.join(out_dir, out_name)
    final_path = out_path[:-3] if unzip and out_name.endswith(".gz") else out_path
    if os.path.exists(final_path):
        return final_path
    url = f"{RCSB_FILES_BASE}/{out_name}"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)
        if unzip and out_path.endswith(".gz"):
            with gzip.open(out_path, "rb") as gz:
                with open(final_path, "wb") as out:
                    shutil.copyfileobj(gz, out)
            os.remove(out_path)
        return final_path if os.path.exists(final_path) else None
    except Exception:
        return None


def download_pdb_by_id(pdb_id: str, out_dir: str, file_type: str = "pdb", unzip: bool = True):
    """Alias for download_rcsb_structure (backward compatibility)."""
    return download_rcsb_structure(pdb_id, out_dir, file_type, unzip)


def _download_one(pdb_id, out_dir, file_type, unzip):
    path = download_rcsb_structure(pdb_id, out_dir, file_type, unzip)
    return pdb_id, "successfully downloaded" if path else "failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run tests for query_rcsb_structure, download_rcsb_structure; output under example/database/rcsb")
    parser.add_argument("-i", "--pdb_id")
    parser.add_argument("-f", "--pdb_id_file")
    parser.add_argument("-o", "--out_dir", default=".")
    parser.add_argument("-t", "--type", default="pdb", choices=list(SUFFIX_MAP))
    parser.add_argument("-u", "--unzip", action="store_true")
    parser.add_argument("-e", "--error_file")
    parser.add_argument("-n", "--num_workers", type=int, default=12)
    args = parser.parse_args()

    if args.test:
        out_base = os.path.join("example", "database", "rcsb", "structure")
        os.makedirs(out_base, exist_ok=True)
        test_id = "4HHB"
        print("Testing query_rcsb_structure(...)")
        text = query_rcsb_structure(test_id, file_type=args.type)
        if text.strip().startswith("{"):
            sample_path = os.path.join(out_base, "query_structure_sample.json")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(text[:2000])
        else:
            sample_path = os.path.join(out_base, "query_structure_sample.txt")
            with open(sample_path, "w", encoding="utf-8") as f:
                f.write(text[:5000] if len(text) > 5000 else text)
        print(f"  saved to {sample_path}")
        print("Testing download_rcsb_structure(...)")
        path = download_rcsb_structure(test_id, out_base, file_type=args.type, unzip=True)
        print(f"  -> {path}")
        print(f"Done. Output under {out_base}")
        exit(0)

    if not args.pdb_id and not args.pdb_id_file:
        print("Error: Must provide either pdb_id or pdb_id_file")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    errors, messages = [], []
    pdbs = [args.pdb_id] if args.pdb_id else [line.strip() for line in open(args.pdb_id_file) if line.strip()]

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = {ex.submit(_download_one, p, args.out_dir, args.type, args.unzip): p for p in pdbs}
        for fut in tqdm(as_completed(futures), total=len(pdbs), desc="Downloading"):
            pdb_id, msg = fut.result()
            if "failed" in msg:
                errors.append(pdb_id)
                messages.append(msg)

    if errors and args.error_file:
        d = os.path.dirname(args.error_file)
        os.makedirs(d, exist_ok=True)
        pd.DataFrame({"protein": errors, "message": messages}).to_csv(args.error_file, index=False)

    n_total = len(pdbs)
    print(f"Done. Output: {args.out_dir} | Total: {n_total}, OK: {n_total - len(errors)}, Failed: {len(errors)}")
