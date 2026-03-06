"""
InterPro entry/family metadata: query and download by InterPro ID only.
Returns entry info (name, type, description, etc.) from InterPro API.
"""
import json
import os
import argparse
import requests

INTERPRO_ENTRY_BASE = "https://www.ebi.ac.uk/interpro/api/entry/interpro"


def query_interpro_metadata(interpro_id: str) -> str:
    """Query InterPro entry/family metadata by InterPro ID. Returns JSON string. No file save."""
    url = f"{INTERPRO_ENTRY_BASE}/{interpro_id}"
    try:
        response = requests.get(url, timeout=120)
        if response.status_code == 404:
            return json.dumps({"success": False, "interpro_id": interpro_id, "error": "Not found"})
        response.raise_for_status()
        data = response.json()
        return json.dumps(data, indent=2, ensure_ascii=False)
    except requests.RequestException as e:
        return json.dumps({"success": False, "interpro_id": interpro_id, "error": str(e)})


def download_interpro_metadata(interpro_id: str, out_dir: str) -> str:
    """Download InterPro entry metadata by InterPro ID. Saves JSON file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{interpro_id}.json")
    text = query_interpro_metadata(interpro_id)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return f"{interpro_id} failed: {data.get('error', 'unknown')}"
    except json.JSONDecodeError:
        pass
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return f"{interpro_id} metadata downloaded"


if __name__ == "__main__":
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(description="InterPro entry metadata: query/download by InterPro ID")
    parser.add_argument("--test", action="store_true", help="Run tests for query_interpro_metadata, download_interpro_metadata; output under example/database/interpro")
    parser.add_argument("-i", "--interpro_id", type=str, help="Single InterPro ID")
    parser.add_argument("-f", "--interpro_id_file", type=str, help="Input file containing a list of InterPro IDs")
    parser.add_argument("-o", "--out_dir", type=str, help="Output directory to save JSON files")
    parser.add_argument("-e", "--error_file", type=str, help="File to record failed IDs")
    parser.add_argument("-n", "--num_workers", type=int, default=12)
    args = parser.parse_args()

    if args.test:
        out_dir = os.path.join("example", "database", "interpro", "metadata")
        os.makedirs(out_dir, exist_ok=True)
        test_id = "IPR001557"
        print("Testing query_interpro_metadata(...)")
        text = query_interpro_metadata(test_id)
        sample_path = os.path.join(out_dir, "query_metadata_sample.json")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  saved to {sample_path}")
        print("Testing download_interpro_metadata(...)")
        msg = download_interpro_metadata(test_id, out_dir)
        print(f"  {msg}")
        print(f"Done. Output under {out_dir}")
        exit(0)

    if not args.out_dir:
        print("Error: -o/--out_dir required (or use --test)")
        exit(1)
    if not args.interpro_id and not args.interpro_id_file:
        print("Error: Must provide either interpro_id or interpro_id_file")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    ids = [args.interpro_id] if args.interpro_id else [line.strip() for line in open(args.interpro_id_file) if line.strip()]
    errors, messages = [], []

    def _dl(iid):
        return iid, download_interpro_metadata(iid, args.out_dir)

    if len(ids) == 1:
        iid, msg = _dl(ids[0])
        print(msg)
        if "failed" in msg:
            errors.append(iid)
            messages.append(msg)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futs = {ex.submit(_dl, iid): iid for iid in ids}
            for fut in tqdm(as_completed(futs), total=len(ids), desc="Downloading"):
                iid, msg = fut.result()
                if "failed" in msg:
                    errors.append(iid)
                    messages.append(msg)

    if errors and args.error_file:
        d = os.path.dirname(args.error_file)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(args.error_file, "w") as f:
            for p, m in zip(errors, messages):
                f.write(f"{p} - {m}\n")

    n_total = len(ids)
    print(f"Done. Output: {args.out_dir} | Total: {n_total}, OK: {n_total - len(errors)}, Failed: {len(errors)}")
