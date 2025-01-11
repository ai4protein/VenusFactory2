import os
import json
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from esm.models.vqvae import StructureTokenEncoder
from get_esm3_structure_seq import get_esm3_structure_seq
from get_foldseek_structure_seq import get_foldseek_structure_seq
from get_secondary_structure_seq import get_secondary_structure_seq

# ignore the warning
import warnings
warnings.filterwarnings("ignore")

def ESM3_structure_encoder_v0(device: torch.device | str = "cpu"):
    model = (
        StructureTokenEncoder(
            d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096
        )
        .to(device)
        .eval()
    )
    state_dict = torch.load(
        "./src/data/weight/esm3_structure_encoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, default='dataset/sesadapter/DeepET/esmfold_pdb')
    parser.add_argument("--pdb_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default='dataset/sesadapter/DeepET')
    parser.add_argument("--merge_into", type=str, default='csv', choices=['json', 'csv'])
    args = parser.parse_args()

    device = "cuda:0"
    esm3_encoder = ESM3_structure_encoder_v0(device)
    
    if args.pdb_dir is not None:
        dir_name = os.path.basename(args.pdb_dir)
        pdb_files = os.listdir(args.pdb_dir)
        ss_results, esm3_results = [], []
        for pdb_file in tqdm(pdb_files):
            ss_result, error = get_secondary_structure_seq(os.path.join(args.pdb_dir, pdb_file))
            if error is not None:
                print(error)
                continue
            ss_results.append(ss_result)
            esm3_result = get_esm3_structure_seq(os.path.join(args.pdb_dir, pdb_file), esm3_encoder, device)
            esm3_results.append(esm3_result)
            # clear cuda cache
            torch.cuda.empty_cache()
        with open(os.path.join(args.out_dir, f"{dir_name}_ss.json"), "w") as f:
            f.write("\n".join([json.dumps(r) for r in ss_results]))
        with open(os.path.join(args.out_dir, f"{dir_name}_esm3.json"), "w") as f:
            f.write("\n".join([json.dumps(r) for r in esm3_results]))
        
        fs_results = get_foldseek_structure_seq(args.pdb_dir)
        with open(os.path.join(args.out_dir, f"{dir_name}_fs.json"), "w") as f:
            f.write("\n".join([json.dumps(r) for r in fs_results]))
    
        if args.merge_into == 'csv':
            # read json files and merge to a single csv according to the same 'name' column
            ss_json = os.path.join(args.out_dir, f"{dir_name}_ss.json")
            esm3_json = os.path.join(args.out_dir, f"{dir_name}_esm3.json")
            fs_json = os.path.join(args.out_dir, f"{dir_name}_fs.json")
            # load json line files
            ss_df = pd.read_json(ss_json, lines=True)
            esm3_df = pd.read_json(esm3_json, lines=True)
            fs_df = pd.read_json(fs_json, lines=True)
            # merge the three dataframes by the 'name' column
            df = pd.merge(ss_df, fs_df, on='name', how='inner')
            df = pd.merge(df, esm3_df, on='name', how='inner')
            df.to_csv(os.path.join(args.out_dir, f"{dir_name}.csv"), index=False)
