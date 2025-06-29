import random
import torch
import os
import biotite
import torch
import numpy as np
import torch.utils.data as data
from torch_geometric.data import Data, Batch
from tqdm import tqdm
from typing import List
from biotite.structure.residues import get_residues
from biotite.sequence import ProteinSequence
from biotite.structure.io import pdbx, pdb
from biotite.structure import filter_backbone
from biotite.structure import get_chains

def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain] 
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in structure]
    structure = structure[chain_filter]
    return structure

def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """
    def filterfn(s, axis=None):
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        sum = filters.sum(0)
        if not np.all(sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)

def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates
            - seq is the extracted sequence
    """
    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return coords

def extract_seq_from_pdb(pdb_file, chain=None):
    """
    Args:
        structure: An instance of biotite AtomArray
    Returns:
        - seq is the extracted sequence
    """
    structure = load_structure(pdb_file, chain)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return seq

def convert_graph(graph):
    graph = Data(
        node_s=graph.node_s.to(torch.float32),
        node_v=graph.node_v.to(torch.float32),
        edge_index=graph.edge_index.to(torch.int64),
        edge_s=graph.edge_s.to(torch.float32),
        edge_v=graph.edge_v.to(torch.float32),
        )
    return graph


def collate_fn(batch):
    data_list_1 = []
    data_list_2 = []
    labels = []
    
    for item in batch:
        data_list_1.append(item[0])
        data_list_2.append(item[1])
        labels.append(item[2])
    
    batch_1 = Batch.from_data_list(data_list_1)
    batch_2 = Batch.from_data_list(data_list_2)
    labels = torch.tensor(labels, dtype=torch.float)
    return (batch_1, batch_2, labels)


class ProteinGraphDataset(data.Dataset):
    """
    args:
        data_list: list of Data
        extra_return: list of extra return data name
    
    """
    def __init__(self, data_list, extra_return=None):
        super(ProteinGraphDataset, self).__init__()
        
        self.data_list = data_list
        self.node_counts = [e.node_s.shape[0] for e in data_list]
        self.extra_return = extra_return
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, i):
        graph = self.data_list[i]
        # RuntimeError: "LayerNormKernelImpl" not implemented for 'Long'
        graph = Data(
            node_s=torch.as_tensor(graph.node_s, dtype=torch.float32), 
            node_v=torch.as_tensor(graph.node_v, dtype=torch.float32),
            edge_index=graph.edge_index, 
            edge_s=torch.as_tensor(graph.edge_s, dtype=torch.float32),
            edge_v=torch.as_tensor(graph.edge_v, dtype=torch.float32)
            )
        if self.extra_return:
            for extra in self.extra_return:
                graph[extra] = self.data_list[i][extra]
        return graph


class BatchSampler(data.Sampler):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design.
    
    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_batch_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, node_counts, max_batch_nodes=3000, shuffle=True):
        
        self.node_counts = node_counts
        self.idx = [i for i in range(len(node_counts)) if node_counts[i] <= max_batch_nodes]
        self.shuffle = shuffle
        self.max_batch_nodes = max_batch_nodes
        self._form_batches()
    
    def _form_batches(self):
        self.batches = []
        if self.shuffle: random.shuffle(self.idx)
        idx = self.idx
        while idx:
            batch = []
            n_nodes = 0
            while idx and n_nodes + self.node_counts[idx[0]] <= self.max_batch_nodes:
                next_idx, idx = idx[0], idx[1:]
                n_nodes += self.node_counts[next_idx]
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch
