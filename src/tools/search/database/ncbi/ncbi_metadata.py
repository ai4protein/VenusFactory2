"""
NCBI metadata: query (return GenBank/XML text) and download (save to file).
"""
import json
import os
import time
import argparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


def query_ncbi_meta(ncbi_id: str, db: str = "protein", rettype: str = "gb") -> str:
    """Query NCBI entry metadata. Returns GenBank/XML text or error JSON string. No file save."""
    retmode = "text" if rettype == "gb" else "xml"
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={db}&id={ncbi_id}&rettype={rettype}&retmode={retmode}"
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


def download_ncbi_meta(ncbi_id: str, out_dir: str, db: str = "protein", rettype: str = "gb") -> str:
    """Download NCBI metadata by accession. Saves .gb or .xml file. Returns message string."""
    os.makedirs(out_dir, exist_ok=True)
    ext = "gb" if rettype == "gb" else "xml"
    out_path = os.path.join(out_dir, f"{ncbi_id}.{ext}")
    if os.path.exists(out_path):
        return f"Skipping {ncbi_id}, already exists"
    text = query_ncbi_meta(ncbi_id, db, rettype)
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("success") is False:
            return f"{ncbi_id} failed: {data.get('error', 'unknown')}"
    except json.JSONDecodeError:
        pass
    with open(out_path, "w") as f:
        f.write(text)
    return f"{ncbi_id} meta downloaded"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run tests for query_ncbi_meta, download_ncbi_meta; output under example/database/ncbi")
    parser.add_argument("-i", "--id", dest="ncbi_id")
    parser.add_argument("-f", "--file")
    parser.add_argument("-o", "--out_dir")
    parser.add_argument("-d", "--db", default="protein")
    parser.add_argument("-t", "--rettype", default="gb", choices=["gb", "xml"])
    parser.add_argument("-n", "--num_workers", type=int, default=10)
    parser.add_argument("-e", "--error_file")
    args = parser.parse_args()

    if args.test:
        out_base = os.path.join("example", "database", "ncbi", "metadata")
        os.makedirs(out_base, exist_ok=True)
        meta_dir = os.path.join(out_base, "metadata")
        test_id = "NP_000483.1"
        print("Testing query_ncbi_meta(...)")
        text = query_ncbi_meta(test_id, db=args.db, rettype=args.rettype)
        sample_path = os.path.join(out_base, "query_meta_sample." + ("gb" if args.rettype == "gb" else "xml"))
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(text[:5000] if len(text) > 5000 else text)
        print(f"  saved to {sample_path}")
        print("Testing download_ncbi_meta(...)")
        msg = download_ncbi_meta(test_id, meta_dir, db=args.db, rettype=args.rettype)
        print(f"  {msg}")
        print(f"Done. Output under {out_base}")
        exit(0)

    if not args.out_dir:
        print("Error: -o/--out_dir required (or use --test)")
        exit(1)
    if not args.ncbi_id and not args.file:
        print("Error: Must provide either ncbi_id or file")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    errors, messages = [], []
    ids = [args.ncbi_id] if args.ncbi_id else [line.strip() for line in open(args.file) if line.strip()]

    def _dl(nid):
        return nid, download_ncbi_meta(nid, args.out_dir, args.db, args.rettype)

    if len(ids) == 1:
        nid, msg = _dl(ids[0])
        print(msg)
        if "failed" in msg:
            errors.append(nid)
            messages.append(msg)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {ex.submit(_dl, nid): nid for nid in ids}
            for fut in tqdm(as_completed(futures), total=len(ids), desc="Downloading"):
                nid, msg = fut.result()
                if "failed" in msg:
                    errors.append(nid)
                    messages.append(msg)

    if errors and args.error_file:
        d = os.path.dirname(args.error_file)
        os.makedirs(d, exist_ok=True)
        with open(args.error_file, "w") as f:
            for p, m in zip(errors, messages):
                f.write(f"{p} - {m}\n")

    n_total = len(ids)
    print(f"Done. Output: {args.out_dir} | Total: {n_total}, OK: {n_total - len(errors)}, Failed: {len(errors)}")
