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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--interpro_id", type=str, default="IPR000001")
    parser.add_argument("--interpro_json", type=str, default="data/interpro_domain/interpro.json")
    parser.add_argument("--out_dir", type=str, default="data/interpro_domain")
    parser.add_argument("--chunk_num", type=int, default=None)
    parser.add_argument("--chunk_id", type=int, default=None)
    args = parser.parse_args()
    
    if args.interpro_json:
        with open(args.interpro_json, 'r') as f:
            all_data = json.load(f)
            if args.chunk_num is not None and args.chunk_id is not None:
                start, end = args.chunk_id * len(all_data) // args.chunk_num, (args.chunk_id + 1) * len(all_data) // args.chunk_num
                all_data = all_data[start:end]
        
        for data in tqdm(all_data):
            interpro_id = data["metadata"]["accession"]
            interpro_dir = os.path.join(args.out_dir, interpro_id)
            os.makedirs(interpro_dir, exist_ok=True)
            
            start_url = f"https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/{interpro_id}/?extra_fields=counters&page_size=20"
            
            file = os.path.join(interpro_dir, "detail.json")
            if os.path.exists(file):
                continue
            info_data = []
            try:
                info_data = fetch_info_data(start_url)
            except:
                print("Error: ", interpro_id)
                continue
            if info_data == []:
                continue
            with open(file, 'w') as f:
                json.dump(info_data, f)
            
            data["num_proteins"] = len(info_data)
            with open(os.path.join(interpro_dir, "meta.json"), 'w') as f:
                json.dump(data, f)
            
            uids = [d["metadata"]["accession"] for d in info_data]
            with open(os.path.join(interpro_dir, "uids.txt"), 'w') as f:
                f.write("\n".join(uids))
