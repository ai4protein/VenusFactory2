import torch
import os
import sys
sys.path.append(os.getcwd())
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from biotite.structure.io.pdb import PDBFile
from torch.nn import functional as F
from esm.utils.structure.protein_chain import ProteinChain
from esm.utils.constants.esm3 import VQVAE_SPECIAL_TOKENS
from esm.tokenization.structure_tokenizer import StructureTokenizer
from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)

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
        "data/weights/esm3_structure_encoder_v0.pth", map_location=device
    )
    model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_file", type=str, default=None)
    parser.add_argument("--pdb_dir", type=str, default=None)
    parser.add_argument("--out_file", type=str, default='structure_tokens.json')
    args = parser.parse_args()
    
    device="cuda:0"
    results = []
    # result_dict = {'name':[], 'aa_seq':[], 'esm3_structure_tokens':[], 'plddt':[], 'residue_index':[]}
    
    encoder = ESM3_structure_encoder_v0(device)
    
    if args.pdb_file is not None:
        # Extract Unique Chain IDs
        chain_ids = np.unique(PDBFile.read(args.pdb_file).get_structure().chain_id)
        # print(chain_ids)
        # ['L', 'H']

        # By Default, ProteinChain takes first one
        chain = ProteinChain.from_pdb(args.pdb_file, chain_id=chain_ids[0])
        sequence = chain.sequence

        # Encoder
        coords, plddt, residue_index = chain.to_structure_encoder_inputs()
        coords = coords.to(device)
        #plddt = plddt.cuda()
        residue_index = residue_index.to(device)
        _, structure_tokens = encoder.encode(coords, residue_index=residue_index)

        result = {'name':args.pdb_file, 'aa_seq':sequence, 'esm3_structure_seq':structure_tokens.cpu().numpy().tolist()[0]}
        results.append(result)
        
        with open(args.out_file, "w") as f:
            f.write("\n".join([json.dumps(r) for r in results]))
    
    elif args.pdb_dir is not None:
        pdb_files = os.listdir(args.pdb_dir)
        for pdb_file in tqdm(pdb_files):
            # Extract Unique Chain IDs
            chain_ids = np.unique(PDBFile.read(os.path.join(args.pdb_dir, pdb_file)).get_structure().chain_id)
            # print(chain_ids)
            # ['L', 'H']

            # By Default, ProteinChain takes first one
            chain = ProteinChain.from_pdb(os.path.join(args.pdb_dir, pdb_file), chain_id=chain_ids[0])
            sequence = chain.sequence

            # Encoder
            coords, plddt, residue_index = chain.to_structure_encoder_inputs()
            coords = coords.to(device)
            #plddt = pldt.cuda()
            residue_index = residue_index.to(device)
            _, structure_tokens = encoder.encode(coords, residue_index=residue_index)

            result = {'name':pdb_file, 'aa_seq':sequence, 'esm3_structure_seq':structure_tokens.cpu().numpy().tolist()[0]}
            results.append(result)
            
        with open(args.out_file, "w") as f:
            f.write("\n".join([json.dumps(r) for r in results]))
