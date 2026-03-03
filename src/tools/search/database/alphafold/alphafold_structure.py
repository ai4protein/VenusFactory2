"""Download AlphaFold structure (PDB/mmCIF) by UniProt ID. Aligned with AlphaFold DB API v4 (see skill references)."""
import requests
import os
import time
import random
import argparse
import pandas as pd
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Literal

BASE_URL = "https://alphafold.ebi.ac.uk/files/"
DEFAULT_VERSION = "v6"
SUPPORTED_FORMATS = ("pdb", "cif")


def _structure_url(uniprot_id: str, fragment: int = 1, version: str = DEFAULT_VERSION, fmt: str = "pdb") -> str:
    """Build AlphaFold structure file URL. entry_id pattern: AF-{uniprot_id}-F{fragment}."""
    ext = "pdb" if fmt == "pdb" else "cif"
    entry_suffix = f"AF-{uniprot_id}-F{fragment}-model_{version}.{ext}"
    return BASE_URL + entry_suffix


def _fetch_and_save_alphafold_structure(
    uniprot_id: str,
    out_dir: str,
    fmt: str = "pdb",
    version: str = DEFAULT_VERSION,
    fragment: int = 1,
) -> Tuple[bool, Optional[str], str]:
    """
    Fetch AlphaFold structure for one UniProt ID and save to out_dir.
    fmt: 'pdb' or 'cif' (mmCIF). version: e.g. 'v4'. fragment: 1-based fragment index.
    Returns (success, path_or_none, message).
    """
    os.makedirs(out_dir, exist_ok=True)
    ext = "pdb" if fmt == "pdb" else "cif"
    out_path = os.path.join(out_dir, f"{uniprot_id}.{ext}")
    if os.path.exists(out_path):
        return True, out_path, f"{out_path} already exists, skipping"

    url = _structure_url(uniprot_id, fragment=fragment, version=version, fmt=fmt)
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        ua = UserAgent()
        response = session.get(url, headers={"User-Agent": ua.random})
        response.raise_for_status()
        mode = "wb" if fmt == "pdb" else "w"
        encoding = None if fmt == "pdb" else "utf-8"
        with open(out_path, mode, encoding=encoding) as f:
            if fmt == "pdb":
                f.write(response.content)
            else:
                f.write(response.text)
    except Exception as e:
        return False, None, f"{uniprot_id} failed, {e}"

    if random.uniform(0, 1) < 0.2:
        time.sleep(random.uniform(1, 2))
    return True, out_path, f"{uniprot_id} successfully downloaded"


def download_alphafold_structure(
    uniprot_id: str,
    out_dir: str,
    format: Literal["pdb", "cif"] = "pdb",
    version: str = DEFAULT_VERSION,
    fragment: int = 1,
) -> Tuple[bool, Optional[str]]:
    """Download one AlphaFold structure by UniProt ID. Returns (success, path_to_file or None)."""
    success, path, _ = _fetch_and_save_alphafold_structure(
        uniprot_id, out_dir, fmt=format, version=version, fragment=fragment
    )
    return success, path


def query_alphafold_structure(
    uniprot_id: str,
    format: Literal["pdb", "cif"] = "pdb",
    version: str = DEFAULT_VERSION,
    fragment: int = 1,
) -> str:
    """Query AlphaFold structure by UniProt ID. Returns PDB/mmCIF text or error message. No file save."""
    url = _structure_url(uniprot_id, fragment=fragment, version=version, fmt=format)
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        ua = UserAgent()
        response = session.get(url, headers={"User-Agent": ua.random})
        response.raise_for_status()
        return response.text
    except Exception as e:
        return f"{uniprot_id} failed, {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download structure files from AlphaFold DB (API v4).")
    parser.add_argument("--test", action="store_true", help="Run tests for all non-helper functions (query_alphafold_structure, download_alphafold_structure); output under example/database/alphafold")
    parser.add_argument("-i", "--uniprot_id", help="Single UniProt ID to download")
    parser.add_argument("-f", "--uniprot_id_file", type=str, help="Input file containing a list of UniProt ids")
    parser.add_argument("-o", "--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("--format", type=str, default="pdb", choices=["pdb", "cif"], help="Structure format: pdb or cif (mmCIF)")
    parser.add_argument("--version", type=str, default=DEFAULT_VERSION, choices=["v1", "v2", "v4", "v6"], help="AlphaFold DB version (e.g. v4)")
    parser.add_argument("--fragment", type=int, default=1, help="Fragment index (1-based, for multi-fragment proteins)")
    parser.add_argument("-e", "--error_file", type=str, default=None, help="File to store names of proteins that failed to download")
    parser.add_argument("-l", "--index_level", type=int, default=0, help="Build an index of the downloaded files")
    parser.add_argument("-n", "--num_workers", type=int, default=12, help="Number of workers to use for downloading")
    args = parser.parse_args()

    if args.test:
        # Test all executable non-helper functions; output under example/database
        test_id = "A0A1B0GTW7"
        out_base = os.path.join("example", "database", "alphafold")
        structure_dir = os.path.join(out_base, "structure")
        os.makedirs(structure_dir, exist_ok=True)
        print("Testing query_alphafold_structure(...)")
        query_text = query_alphafold_structure(test_id, format=args.format, version=args.version, fragment=args.fragment)
        if len(query_text) > 800:
            print(f"  (first 800 chars): {query_text[:800]}...")
        else:
            print(f"  result: {query_text}")
        query_sample_path = os.path.join(out_base, "query_structure_sample.txt")
        with open(query_sample_path, "w", encoding="utf-8") as f:
            f.write(query_text)
        print(f"  saved full result to {query_sample_path}")
        print("Testing download_alphafold_structure(...)")
        ok, path = download_alphafold_structure(test_id, structure_dir, format=args.format, version=args.version, fragment=args.fragment)
        print(f"  success={ok}, path={path}")
        print(f"Done. Output under {out_base}")
        exit(0)

    if not args.uniprot_id and not args.uniprot_id_file:
        print("Error: Must provide either uniprot_id or uniprot_id_file")
        exit(1)

    error_proteins = []
    error_messages = []

    def download_af_structure(uniprot_id, args):
        out_dir = args.out_dir
        for index in range(args.index_level):
            index_dir_name = "".join(list(uniprot_id)[: index + 1])
            out_dir = os.path.join(out_dir, index_dir_name)
        _, __, message = _fetch_and_save_alphafold_structure(
            uniprot_id, out_dir, fmt=args.format, version=args.version, fragment=args.fragment
        )
        return uniprot_id, message

    if args.uniprot_id:
        uniprot_id, message = download_af_structure(args.uniprot_id, args)
        print(message)
        if "failed" in message:
            error_proteins.append(uniprot_id)
            error_messages.append(message)

    elif args.uniprot_id_file:
        with open(args.uniprot_id_file) as f:
            pdbs = [line.strip() for line in f if line.strip()]

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_pdb = {executor.submit(download_af_structure, pdb, args): pdb for pdb in pdbs}

            with tqdm(total=len(pdbs), desc="Downloading Files") as bar:
                for future in as_completed(future_to_pdb):
                    pdb, message = future.result()
                    bar.set_description(message)
                    if "failed" in message:
                        error_proteins.append(pdb)
                        error_messages.append(message)
                    bar.update(1)

    if args.error_file and error_proteins:
        error_dict = {"protein": error_proteins, "error": error_messages}
        error_dir = os.path.dirname(args.error_file)
        os.makedirs(error_dir, exist_ok=True)
        pd.DataFrame(error_dict).to_csv(args.error_file, index=False)

    n_total = 1 if args.uniprot_id else len(pdbs)
    print(f"Done. Output: {args.out_dir} | Total: {n_total}, OK: {n_total - len(error_proteins)}, Failed: {len(error_proteins)}")
