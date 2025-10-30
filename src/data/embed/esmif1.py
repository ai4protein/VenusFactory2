import esm
import torch
import os
import sys
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
from esm.inverse_folding.util import CoordBatchConverter
from esm import pretrained

def esmif1_embed_single(model_name_or_path, pdb_file, pooling=None):
    model, alphabet = pretrained.load_model_and_alphabet(model_name_or_path)
    model.cuda()
    model.eval()
    batch_converter = CoordBatchConverter(alphabet)
    coords, sequence, _ = parse_PDB(pdb_file)
    coords = {
        'N': coords[:, 0],
        'CA': coords[:, 1],
        'C': coords[:, 2]
    }
    dist, omega, theta, phi = process_coords(coords)
    batch = [(torch.tensor(dist, dtype=torch.float).cuda(), None, torch.tensor(omega, dtype=torch.float).cuda(), torch.tensor(theta, dtype=torch.float).cuda(), torch.tensor(phi, dtype=torch.float).cuda())]
    src, nodes, edges, connections, edge_mask = collater(batch)
    rep = model(src.cuda(), nodes.cuda(), edges.cuda(), connections.cuda(), edge_mask.cuda(), result='repr')[0]
    if pooling == 'mean':
        pooled_features = rep.mean(dim=0).cpu().detach().numpy()
    elif pooling == 'max':
        pooled_features = rep.max(dim=0).cpu().detach().numpy()
    elif pooling == 'sum':
        pooled_features = rep.sum(dim=0).cpu().detach().numpy()
    else:
        pooled_features = rep.detach().cpu().numpy()
    return pooled_features
    

def esmif1_embed(model_name_or_path, pdb_path, embed_type, chunck_id, output_dir):
    pdbs = sorted(os.listdir(pdb_path))[chunck_id*100000: (chunck_id+1)*100000]
    pdb_infos = {}
    model, alphabet = pretrained.load_model_and_alphabet(model_name_or_path)
    model.cuda()
    model.eval()
    batch_converter = CoordBatchConverter(alphabet)
    
    for pdb in tqdm(pdbs):
        single_pdb_path = os.path.join(pdb_path, pdb)
        coords, pdb_seq = esm.inverse_folding.util.load_coords(single_pdb_path, "A")
        batch = [(coords, None, pdb_seq)]
        coords_, confidence, strs, tokens, padding_mask = batch_converter(batch)
        prev_output_tokens = tokens[:, :-1]
        hidden_states, _ = model.forward(
            coords_.cuda(),
            padding_mask.cuda(),
            confidence.cuda(),
            prev_output_tokens.cuda(),
            features_only=True,
        )
        # last_hidden_state: [1, 512, 1]
        if embed_type == 'last_hidden_state':
            last_hidden_state = hidden_states[0,:,-1]
        elif embed_type == 'mean_hidden_state':
            last_hidden_state = hidden_states[0,:,:].mean(dim=1)
        pdb_infos[pdb] = {"embedding": last_hidden_state.cpu().detach().numpy(), "seq": pdb_seq}

    # save embedding
    out_file = os.path.join(output_dir, f'{model_name_or_path.split('/')[-1]}_{embed_type}_chunk{chunck_id}.pt')
    torch.save(pdb_infos, out_file)

def create_args():
    parser = argparse.ArgumentParser(description='Extract embeddings from ESM_if')
    parser.add_argument('--model_name_or_path', type=str, default='esm_if1_gvp4_t16_142M_UR50', help='model name or path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--pdb_dir', type=str, default='/home/user4/data/swiss_prot_pdb/', help='path to pdb directory')
    parser.add_argument('--output_dir', type=str, default='dataset/database/embed', help='path to output file')
    parser.add_argument('--embed_type', type=str, default='last_hidden_state', help='last_hidden_state or mean_hidden_state')
    parser.add_argument('--chunck_id', type=int, default=0, help='chunck_id id')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_args()
    esmif1_embed(args.model_name_or_path, args.pdb_dir, args.embed_type, args.chunck_id, args.output_dir)