"""Download AlphaFold structure (PDB) by UniProt ID."""
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
from typing import Optional, Tuple

BASE_URL = "https://alphafold.ebi.ac.uk/files/AF-"


def _fetch_and_save_alphafold_pdb(uniprot_id: str, out_dir: str) -> Tuple[bool, Optional[str], str]:
    """
    Fetch AlphaFold PDB for one UniProt ID and save to out_dir.
    Returns (success, path_or_none, message).
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{uniprot_id}.pdb")
    if os.path.exists(out_path):
        return True, out_path, f"{out_path} already exists, skipping"

    url = BASE_URL + uniprot_id + "-F1-model_v6.pdb"
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        ua = UserAgent()
        response = session.get(url, headers={"User-Agent": ua.random})
        response.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(response.content)
    except Exception as e:
        return False, None, f"{uniprot_id} failed, {e}"

    if random.uniform(0, 1) < 0.2:
        time.sleep(random.uniform(1, 2))
    return True, out_path, f"{uniprot_id} successfully downloaded"


def download_alphafold_structure(uniprot_id: str, out_dir: str) -> Tuple[bool, Optional[str]]:
    """Download one AlphaFold structure by UniProt ID. Returns (success, path_to_pdb or None)."""
    success, path, _ = _fetch_and_save_alphafold_pdb(uniprot_id, out_dir)
    return success, path

def query_alphafold_structure(uniprot_id: str) -> str:
    """Query AlphaFold structure by UniProt ID. Returns PDB text or error message string. No file save."""
    url = BASE_URL + uniprot_id + "-F1-model_v6.pdb"
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
    parser = argparse.ArgumentParser(description="Download files from AlphaFold.")
    parser.add_argument("-i", "--uniprot_id", help="Single UniProt ID to download")
    parser.add_argument("-f", "--uniprot_id_file", type=str, help="Input file containing a list of UniProt ids")
    parser.add_argument("-o", "--out_dir", type=str, default=".", help="Output directory")
    parser.add_argument("-e", "--error_file", type=str, default=None, help="File to store names of proteins that failed to download")
    parser.add_argument("-l", "--index_level", type=int, default=0, help="Build an index of the downloaded files")
    parser.add_argument("-n", "--num_workers", type=int, default=12, help="Number of workers to use for downloading")
    args = parser.parse_args()

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
        _, __, message = _fetch_and_save_alphafold_pdb(uniprot_id, out_dir)
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
