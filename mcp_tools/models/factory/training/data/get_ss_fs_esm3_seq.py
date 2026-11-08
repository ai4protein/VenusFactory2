import os
import json
import argparse
import pandas as pd
from tqdm import tqdm
from mcp_tools.models.factory.training.data.esm3_runtime import encode_structures
from mcp_tools.models.factory.training.data.get_foldseek_structure_seq import get_foldseek_structure_seq
from mcp_tools.models.factory.training.data.get_secondary_structure_seq import get_secondary_structure_seq
from mcp_tools.models.factory.training.data.get_prosst_str_token import convert_predict_results

# ignore the warning
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, default='dataset/sesadapter/DeepET/esmfold_pdb')
    parser.add_argument("--pdb_file", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default='dataset/sesadapter/DeepET')
    parser.add_argument("--merge_into", type=str, default='csv', choices=['json', 'csv'])
    parser.add_argument("--save_intermediate", action='store_true')
    args = parser.parse_args()

    device = "cuda:0"
    
    if args.pdb_dir is not None:
        dir_name = os.path.basename(args.pdb_dir)
        pdb_files = sorted([p for p in os.listdir(args.pdb_dir) if p.endswith('.pdb')])
        ss_results, esm3_inputs = [], []
        for pdb_file in tqdm(pdb_files):
            ss_result, error = get_secondary_structure_seq(os.path.join(args.pdb_dir, pdb_file))
            if error is not None:
                print(error)
                continue
            ss_results.append(ss_result)
            esm3_inputs.append(os.path.join(args.pdb_dir, pdb_file))
        esm3_results = encode_structures(esm3_inputs, device)
        ss_csv = os.path.join(args.out_dir, f"{dir_name}_ss.csv")
        esm3_csv = os.path.join(args.out_dir, f"{dir_name}_esm3.csv")
        fs_csv = os.path.join(args.out_dir, f"{dir_name}_fs.csv")
        prosst_csv = os.path.join(args.out_dir, f"{dir_name}_prosst.csv")

        pd.DataFrame(ss_results).to_csv(ss_csv, index=False)
        esm3_df = pd.DataFrame(esm3_results)
        esm3_df["esm3_structure_seq"] = esm3_df["esm3_structure_seq"].apply(json.dumps)
        esm3_df.to_csv(esm3_csv, index=False)

        fs_results = get_foldseek_structure_seq(args.pdb_dir)
        pd.DataFrame(fs_results).to_csv(fs_csv, index=False)

        from mcp_tools.models.pretrained.prosst.structure.get_sst_seq import SSTPredictor
        processor = SSTPredictor(structure_vocab_size=2048)
        prosst_tokens = convert_predict_results(
            processor.predict_from_pdb([os.path.join(args.pdb_dir, p) for p in pdb_files]), 2048,
        )
        prosst_df = pd.DataFrame(prosst_tokens)
        for col in prosst_df.columns:
            if col != 'name' and col != 'aa_seq' and prosst_df[col].dtype == object:
                prosst_df[col] = prosst_df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
        prosst_df.to_csv(prosst_csv, index=False)

        if args.merge_into == 'csv':
            ss_df = pd.read_csv(ss_csv)
            esm3_df = pd.read_csv(esm3_csv)
            fs_df = pd.read_csv(fs_csv)
            prosst_df = pd.read_csv(prosst_csv)
            df = pd.merge(ss_df, fs_df, on='name', how='inner')
            df = pd.merge(df, esm3_df, on='name', how='inner')
            df = pd.merge(df, prosst_df, on='name', how='inner')
            df = df.sort_values(by='name')
            df.to_csv(os.path.join(args.out_dir, f"{dir_name}.csv"), index=False)

            if not args.save_intermediate:
                for f in [ss_csv, esm3_csv, fs_csv, prosst_csv]:
                    if os.path.exists(f):
                        os.remove(f)
