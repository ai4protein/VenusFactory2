import requests
import os
import time
import random
import argparse
import pandas as pd
from fake_useragent import UserAgent
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def download(pdb, outdir):
    url = BASE_URL + pdb + "-F1-model_v4.pdb"
    out_path = os.path.join(outdir, f"{pdb}.pdb")

    message = f"{pdb} successfully downloaded"
    
    if os.path.exists(out_path):
        return f"{out_path} already exists, skipping"

    # Use a random user agent
    ua = UserAgent()

    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('http://', HTTPAdapter(max_retries=retries))
    session.mount('https://', HTTPAdapter(max_retries=retries))

    try:
        response = session.get(url, headers={'User-Agent': ua.random})
        response.raise_for_status()
        with open(out_path, 'wb') as file:
            file.write(response.content)
    except Exception as e:
        return f"{pdb} failed, {e}"
    
    # Sleep for 1-2 seconds with 20% probability
    if random.uniform(0, 1) < 0.2:
        time.sleep(random.uniform(1, 2))
    return message

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download files from RCSB.')
    parser.add_argument('-f', '--uniprot_id_file', type=str, required=True, help='Input file containing a list of PDB ids')
    parser.add_argument('-o', '--out_dir', type=str, default='.', help='Output directory')
    parser.add_argument('-e', '--error_file', type=str, default=None, help='File to store names of proteins that failed to download')
    parser.add_argument('-i', '--index_level', type=int, default=0, help='Build an index of the downloaded files')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='Number of workers to use for downloading')
    args = parser.parse_args()

    BASE_URL = "https://alphafold.ebi.ac.uk/files/AF-"
    error_proteins = []
    error_messages = []

    pdbs = open(args.uniprot_id_file, 'r').read().splitlines()
    
    def download_pdb(pdb, args):
        out_dir = args.out_dir
        for index in range(args.index_level):
            index_dir_name = "".join(list(pdb)[:index + 1])
            out_dir = os.path.join(out_dir, index_dir_name)
        os.makedirs(out_dir, exist_ok=True)
        message = download(pdb, out_dir)
        return pdb, message
    
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_pdb = {executor.submit(download_pdb, pdb, args): pdb for pdb in pdbs}

        with tqdm(total=len(pdbs), desc="Downloading Files") as bar:
            for future in as_completed(future_to_pdb):
                pdb, message = future.result()
                bar.set_description(message)
                if "failed" in message:
                    error_proteins.append(pdb)
                    error_messages.append(message)
                bar.update(1)

    if args.error_file:
        error_dict = {"protein": error_proteins, "error": error_messages}
        error_dir = os.path.dirname(args.error_file)
        os.makedirs(error_dir, exist_ok=True)
        pd.DataFrame(error_dict).to_csv(args.error_file, index=False)
