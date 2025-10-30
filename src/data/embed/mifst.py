import esm
import torch
import os
import sys
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
from src.mutation.models.sequence_models.pdb_utils import parse_PDB, process_coords
from src.mutation.models.sequence_models.pretrained import load_model_and_alphabet
from src.mutation.models.sequence_models.constants import PROTEIN_ALPHABET

def mifst_embed_single(model_name_or_path, pdb_file):
    model, collater = load_model_and_alphabet(model_name_or_path)
    model.cuda()
    model.eval()
    coords, sequence, _ = parse_PDB(pdb_file)
    coords = {
        'N': coords[:, 0],
        'CA': coords[:, 1],
        'C': coords[:, 2]
    }
    dist, omega, theta, phi = process_coords(coords)
    batch = [[sequence, torch.tensor(dist, dtype=torch.float).cuda(),
                torch.tensor(omega, dtype=torch.float).cuda(),
                torch.tensor(theta, dtype=torch.float).cuda(), 
                torch.tensor(phi, dtype=torch.float).cuda()]]
    src, nodes, edges, connections, edge_mask = collater(batch)
    rep = model(src.cuda(), nodes.cuda(), edges.cuda(), connections.cuda(), edge_mask.cuda(), result='repr')[0]
    rep_mean = rep.mean(dim=0).cpu().detach()
    return rep_mean.detach().cpu().numpy()



def mifst_embed(model_name_or_path, pdb_path, embed_type, chunck_id, output_dir):
    pdbs = sorted(os.listdir(pdb_path))[chunck_id*100000: (chunck_id+1)*100000]
    pdb_infos = {}

    model, collater = load_model_and_alphabet(model_name_or_path)
    model.cuda()
    model.eval()
    
    for pdb in tqdm(pdbs):
        coords, sequence, _ = parse_PDB(os.path.join(pdb_path, pdb))
        coords = {
            'N': coords[:, 0],
            'CA': coords[:, 1],
            'C': coords[:, 2]
        }
        dist, omega, theta, phi = process_coords(coords)
        batch = [[sequence, torch.tensor(dist, dtype=torch.float).cuda(),
                torch.tensor(omega, dtype=torch.float).cuda(),
                torch.tensor(theta, dtype=torch.float).cuda(), 
                torch.tensor(phi, dtype=torch.float).cuda()]]
        src, nodes, edges, connections, edge_mask = collater(batch)
        rep = model(src.cuda(), nodes.cuda(), edges.cuda(), connections.cuda(), edge_mask.cuda(), result='repr')[0]
        rep_mean = rep.mean(dim=0).cpu().detach()
        pdb_infos[pdb] = {"embedding": rep_mean.numpy(), "seq": sequence}
        
        
    # save embedding
    out_file = os.path.join(output_dir, f'{model_name_or_path.split('/')[-1]}_{embed_type}_chunk{chunck_id}.pt')
    torch.save(pdb_infos, out_file)

def create_args():
    parser = argparse.ArgumentParser(description='Extract embeddings from MIF-ST')
    parser.add_argument('--model_name_or_path', type=str, default='mifst', help='model name or path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--pdb_dir', type=str, default='/home/user4/data/swiss_prot_pdb/', help='path to pdb directory')
    parser.add_argument('--output_dir', type=str, default='dataset/database/embed', help='path to output file')
    parser.add_argument('--embed_type', type=str, default='last_hidden_state', help='last_hidden_state or mean_hidden_state')
    parser.add_argument('--chunck_id', type=int, default=0, help='chunck_id id')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_args()
    mifst_embed(args.model_name_or_path, args.pdb_dir, args.embed_type, args.chunck_id, args.output_dir)