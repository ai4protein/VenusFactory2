import requests
import time
import json
import os
import argparse
from tqdm import tqdm

def fetch_info_data(url):
    data_list = []
    while url:
        response = requests.get(url)
        data = response.json()
        data_list.extend(data["results"])
        url = data.get("next")
        time.sleep(10)
    return data_list

def download_single_interpro(interpro_id, out_dir):
    interpro_dir = os.path.join(out_dir, interpro_id)
    os.makedirs(interpro_dir, exist_ok=True)
    
    start_url = f"https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/{interpro_id}/?extra_fields=counters&page_size=20"
    
    file = os.path.join(interpro_dir, "detail.json")
    if os.path.exists(file):
        return f"Skipping {interpro_id}, already exists"
        
    info_data = []
    try:
        info_data = fetch_info_data(start_url)
    except:
        return f"Error downloading {interpro_id}"
    
    if not info_data:
        return f"No data found for {interpro_id}"
        
    with open(file, 'w') as f:
        json.dump(info_data, f)
    
    # Save metadata
    meta_data = {
        "metadata": {"accession": interpro_id},
        "num_proteins": len(info_data)
    }
    with open(os.path.join(interpro_dir, "meta.json"), 'w') as f:
        json.dump(meta_data, f)
    
    # Save UIDs
    uids = [d["metadata"]["accession"] for d in info_data]
    with open(os.path.join(interpro_dir, "uids.txt"), 'w') as f:
        f.write("\n".join(uids))
    
    return f"Successfully downloaded {interpro_id}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpro_id", type=str, default=None)
    parser.add_argument("--interpro_json", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="download/interpro_domain")
    parser.add_argument("--error_file", type=str, default=None)
    parser.add_argument("--chunk_num", type=int, default=None)
    parser.add_argument("--chunk_id", type=int, default=None)
    args = parser.parse_args()
    
    if not args.interpro_id and not args.interpro_json:
        print("Error: Must provide either interpro_id or interpro_json")
        exit(1)
    
    os.makedirs(args.out_dir, exist_ok=True)
    error_proteins = []
    error_messages = []
    
    if args.interpro_id:
        result = download_single_interpro(args.interpro_id, args.out_dir)
        print(result)
        if "Error" in result or "No data" in result:
            error_proteins.append(args.interpro_id)
            error_messages.append(result)
    
    elif args.interpro_json:
        dir_path = os.path.dirname(args.interpro_json)
        os.makedirs(dir_path, exist_ok=True)
        
        try:
            with open(args.interpro_json, 'r') as f:
                all_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Could not find file {args.interpro_json}")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON file {args.interpro_json}")
            exit(1)
            
        if args.chunk_num is not None and args.chunk_id is not None:
            start = args.chunk_id * len(all_data) // args.chunk_num
            end = (args.chunk_id + 1) * len(all_data) // args.chunk_num
            all_data = all_data[start:end]
        
        for data in tqdm(all_data):
            interpro_id = data["metadata"]["accession"]
            result = download_single_interpro(interpro_id, args.out_dir)
            if "Error" in result or "No data" in result:
                error_proteins.append(interpro_id)
                error_messages.append(result)

    if error_proteins and args.error_file:
        error_dict = {"protein": error_proteins, "error": error_messages}
        error_file_dir = os.path.dirname(args.error_file)
        os.makedirs(error_file_dir, exist_ok=True)
        with open(args.error_file, 'w') as f:
            for protein, message in zip(error_proteins, error_messages):
                f.write(f"{protein} - {message}\n")
