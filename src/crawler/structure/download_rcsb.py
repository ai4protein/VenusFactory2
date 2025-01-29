import argparse
import os
import requests
import gzip
import shutil
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

download_type_dict = {
    'cif': 'cif.gz',
    'pdb': 'pdb.gz',
    'pdb1': 'pdb1.gz',
    'xml': 'xml.gz',
    'sf': '-sf.cif.gz',
    'mr': 'mr.gz',
    'mrstr': '_mr.str.gz'
}

BASE_URL = "https://files.rcsb.org/download"

def download_and_unzip(file_name, out_dir, unzip):
    url = f"{BASE_URL}/{file_name}"
    out_path = os.path.join(out_dir, file_name)
    message = f"{file_name} successfully downloaded"
    
    if os.path.exists(out_path):
        message = f"{out_path} already exists, skipping"
        return message
    if unzip and os.path.exists(out_path[:-3]):
        message = f"{out_path[:-3]} already exists, skipping"
        return message

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        if unzip and out_path.endswith('.gz'):
            with gzip.open(out_path, 'rb') as gz_file:
                with open(out_path[:-3], 'wb') as out_file:
                    shutil.copyfileobj(gz_file, out_file)
            os.remove(out_path)
        
    except Exception as e:
        message = f"{file_name} failed, {e}"

    return message
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and optionally unzip files from RCSB.')
    parser.add_argument('-i', '--pdb_id', help='Single PDB ID to download')
    parser.add_argument('-f', '--pdb_id_file', help='Input file containing a list of PDB IDs')
    parser.add_argument('-o', '--out_dir', default='.', help='Output directory')
    parser.add_argument('-t', '--type', default='pdb', choices=['cif', 'pdb', 'pdb1', 'xml', 'sf', 'mr', 'mrstr'], help='File type to download')
    parser.add_argument('-u', '--unzip', action='store_true', help='Unzip the downloaded files')
    parser.add_argument('-e', '--error_file', help='File to write PDB ids that failed to download')
    parser.add_argument('-n', '--num_workers', default=12, type=int, help='Number of workers to use for downloading')
    args = parser.parse_args()

    if not args.pdb_id and not args.pdb_id_file:
        print("Error: Must provide either pdb_id or pdb_id_file")
        exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    error_proteins = []
    error_messages = []

    if args.pdb_id:
        message = download_and_unzip(f"{args.pdb_id}.{download_type_dict[args.type]}", args.out_dir, args.unzip)
        print(message)
        if "failed" in message:
            error_proteins.append(args.pdb_id)
            error_messages.append(message)
    
    elif args.pdb_id_file:
        pdbs = open(args.pdb_id_file, 'r').read().splitlines()

        def download_file(pdb, download_type_dict, args):
            message = download_and_unzip(f"{pdb}.{download_type_dict[args.type]}", args.out_dir, args.unzip)
            return pdb, message
        
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_to_pdb = {executor.submit(download_file, pdb, download_type_dict, args): pdb for pdb in pdbs}

            with tqdm(total=len(pdbs), desc="Downloading Files") as bar:
                for future in as_completed(future_to_pdb):
                    pdb, message = future.result()
                    bar.set_description(message)
                    if "failed" in message:
                        error_proteins.append(pdb)
                        error_messages.append(message)
                    bar.update(1)

    if error_proteins and args.error_file:
        error_dict = {'protein': error_proteins, 'message': error_messages}
        error_file_dir = os.path.dirname(args.error_file)
        os.makedirs(error_file_dir, exist_ok=True)
        pd.DataFrame(error_dict).to_csv(args.error_file, index=False)