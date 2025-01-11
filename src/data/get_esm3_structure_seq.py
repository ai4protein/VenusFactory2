import torch
import os
import sys
sys.path.append(os.getcwd())
import json
import argparse
import numpy as np
from tqdm import tqdm
from biotite.structure.io.pdb import PDBFile
from esm.utils.structure.protein_chain import ProteinChain
from esm.models.vqvae import StructureTokenEncoder

VQVAE_CODEBOOK_SIZE = 4096
VQVAE_SPECIAL_TOKENS = {
    "MASK": VQVAE_CODEBOOK_SIZE,
    "EOS": VQVAE_CODEBOOK_SIZE + 1,
    "BOS": VQVAE_CODEBOOK_SIZE + 2,
    "PAD": VQVAE_CODEBOOK_SIZE + 3,
    "CHAINBREAK": VQVAE_CODEBOOK_SIZE + 4,
}

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

def get_esm3_structure_seq(pdb_file, encoder, device="cuda:0"):
    # Extract Unique Chain IDs
    chain_ids = np.unique(PDBFile.read(pdb_file).get_structure().chain_id)
    # print(chain_ids)
    # ['L', 'H']

    # By Default, ProteinChain takes first one
    chain = ProteinChain.from_pdb(pdb_file, chain_id=chain_ids[0])
    sequence = chain.sequence

    # Encoder
    coords, plddt, residue_index = chain.to_structure_encoder_inputs()
    coords = coords.to(device)
    #plddt = plddt.cuda()
    residue_index = residue_index.to(device)
    _, structure_tokens = encoder.encode(coords, residue_index=residue_index)
    
    result = {'name': pdb_file.split('/')[-1], 'aa_seq':sequence, 'esm3_structure_seq':structure_tokens.cpu().numpy().tolist()[0]}
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", type=str, default=None)
    parser.add_argument("--pdb_dir", type=str, default=None)
    parser.add_argument("--out_file", type=str, default='esm3_structure_seq.json')
    args = parser.parse_args()
    
    device="cuda:0"
    results = []
    # result_dict = {'name':[], 'aa_seq':[], 'esm3_structure_seq':[], 'plddt':[], 'residue_index':[]}
    
    encoder = ESM3_structure_encoder_v0(device)
    
    if args.pdb_file is not None:
        result = get_esm3_structure_seq(args.pdb_file, encoder, device)
        results.append(result)
    
    elif args.pdb_dir is not None:
        pdb_files = os.listdir(args.pdb_dir)
        for pdb_file in tqdm(pdb_files):
            result = get_esm3_structure_seq(os.path.join(args.pdb_dir, pdb_file), encoder, device)
            results.append(result)
            
    with open(args.out_file, "w") as f:
        f.write("\n".join([json.dumps(r) for r in results]))
