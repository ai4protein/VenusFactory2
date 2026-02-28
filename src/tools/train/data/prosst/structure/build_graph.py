import torch
import math
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import scipy.spatial as spa
import torch.nn.functional as F
from Bio import PDB
from Bio.SeqUtils import seq1
from torch_geometric.data import Data


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
        )

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _orientations(X_ca):
    forward = _normalize(X_ca[1:] - X_ca[:-1])
    backward = _normalize(X_ca[:-1] - X_ca[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec 

def _positional_embeddings(edge_index, 
                            num_embeddings=16,
                            period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E

    

def generate_graph(pdb_file, max_distance=10):
    """
    generate graph data from pdb file
    
    params:
        pdb_file: pdb file path
        max_distance: cut off
    
    return:
        graph data
    
    """
    pdb_parser = PDB.PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)
    model = structure[0]

    # extract amino acid sequence
    seq = []
    # extract amino acid coordinates
    aa_coords = {"N": [], "CA": [], "C": [], "O": []}
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":
                    seq.append(residue.get_resname())
                    for atom_name in aa_coords.keys():
                        atom = residue[atom_name]
                        aa_coords[atom_name].append(atom.get_coord().tolist())
    aa_seq = "".join([seq1(aa) for aa in seq])
    
        
    # aa means amino acid
    coords = list(zip(aa_coords['N'], aa_coords['CA'], aa_coords['C'], aa_coords['O']))
    coords = torch.tensor(coords)
    # mask out the missing coordinates
    mask = torch.isfinite(coords.sum(dim=(1,2)))
    coords[~mask] = np.inf
    ca_coords = coords[:, 1]
    node_s = torch.zeros(len(ca_coords), 20)
    
    # build graph and max_distance
    distances = spa.distance_matrix(ca_coords, ca_coords)
    edge_index = torch.tensor(np.array(np.where(distances < max_distance)))
    # remove loop
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    # node features
    orientations = _orientations(ca_coords)
    sidechains = _sidechains(coords)
    node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
    
    # edge features
    pos_embeddings = _positional_embeddings(edge_index)
    E_vectors = ca_coords[edge_index[0]] - ca_coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_count=16)
    edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    
    # node_v: [node_num, 3, 3]
    # edge_index: [2, edge_num]
    # edge_s: [edge_num, 16+16]
    # edge_v: [edge_num, 1, 3]
    node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))
    data = Data(
        node_s=node_s, node_v=node_v, 
        edge_index=edge_index, 
        edge_s=edge_s, edge_v=edge_v,
        distances=distances,
        aa_seq=aa_seq,
        ca_coords=ca_coords
    )
    
    return data