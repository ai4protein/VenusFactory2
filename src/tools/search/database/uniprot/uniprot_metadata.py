"""
UniProt metadata: query (return JSON text) and download (save to file).
"""
import json
import os
import argparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

UNIPROT_REST_BASE = "https://rest.uniprot.org/uniprotkb"


def query_uniprot_meta(uniprot_id: str) -> str:
    """Query UniProt entry metadata by ID. Returns JSON text or error JSON string. No file save."""
    url = f"{UNIPROT_REST_BASE}/{uniprot_id}"
    try:
        response = requests.get(url, timeout=30, headers={"Accept": "application/json"})
        if response.status_code == 404:
            return json.dumps({"success": False, "uniprot_id": uniprot_id, "error": "Not found"})
        response.raise_for_status()
        data = response.json()
        return json.dumps(data, indent=2, ensure_ascii=False)
    except requests.RequestException as e:
        return json.dumps({"success": False, "uniprot_id": uniprot_id, "error": str(e)})
    except (ValueError, TypeError) as e:
        return json.dumps({"success": False, "uniprot_id": uniprot_id, "error": str(e)})


def download_uniprot_meta(uniprot_id: str, out_dir: str) -> str:
    """Download UniProt metadata by ID. Saves JSON file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uniprot_id}.json")
    text = query_uniprot_meta(uniprot_id)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False and "error" in data:
            return f"{uniprot_id} failed: {data.get('error', 'unknown')}"
    except json.JSONDecodeError:
        return f"{uniprot_id} failed: invalid response"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return f"{uniprot_id} meta downloaded"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run tests for query_uniprot_meta, download_uniprot_meta; output under example/database/uniprot")
    parser.add_argument("-i", "--uniprot_id")
    parser.add_argument("-f", "--file")
    parser.add_argument("-o", "--out_dir")
    parser.add_argument("-n", "--num_workers", type=int, default=12)
    parser.add_argument("-e", "--error_file")
    args = parser.parse_args()

    if args.test:
        out_base = os.path.join("example", "database", "uniprot", "metadata")
        os.makedirs(out_base, exist_ok=True)
        test_id = "P43403"
        print("Testing query_uniprot_meta(...)")
        text = query_uniprot_meta(test_id)
        sample_path = os.path.join(out_base, "query_meta_sample.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  saved to {sample_path}")
        print("Testing download_uniprot_meta(...)")
        msg = download_uniprot_meta(test_id, out_base)
        print(f"  {msg}")
        print(f"Done. Output under {out_base}")
        exit(0)

    if not args.out_dir:
        print("Error: -o/--out_dir required (or use --test)")
        exit(1)
    if not args.uniprot_id and not args.file:
        print("Error: Must provide either uniprot_id or file")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    errors, messages = [], []
    uids = [args.uniprot_id] if args.uniprot_id else [line.strip() for line in open(args.file) if line.strip()]

    if len(uids) == 1:
        msg = download_uniprot_meta(uids[0], args.out_dir)
        print(msg)
        if "failed" in msg:
            errors.append(uids[0])
            messages.append(msg)
    else:

        def _dl(uid):
            return uid, download_uniprot_meta(uid, args.out_dir)

        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(_dl, uid): uid for uid in uids}
            for fut in tqdm(as_completed(futures), total=len(uids), desc="Downloading"):
                uid, msg = fut.result()
                if "failed" in msg:
                    errors.append(uid)
                    messages.append(msg)

    if errors and args.error_file:
        d = os.path.dirname(args.error_file)
        os.makedirs(d, exist_ok=True)
        with open(args.error_file, "w") as f:
            for p, m in zip(errors, messages):
                f.write(f"{p} - {m}\n")

    n_total = len(uids)
    print(f"Done. Output: {args.out_dir} | Total: {n_total}, OK: {n_total - len(errors)}, Failed: {len(errors)}")
