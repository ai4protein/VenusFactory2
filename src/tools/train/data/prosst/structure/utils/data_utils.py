import random
import torch
import torch.utils.data as data
from torch_geometric.data import Data, Batch
from Bio.SeqIO import PdbIO
import os

def extract_seq_from_pdb(pdb_file: str, chain_id='A'):
    """
    extract sequence from pdb file
    
    Args:
        pdb_file (str): path to the pdb file
        chain_id (str): chain id
    
    Returns:
        dict: a dictionary containing the sequence from SEQRES and ATOM records
    """
    if not os.path.exists(pdb_file):
        print(f"Error: file does not exist at path {pdb_file}")
        return None
        
    pdb_id = os.path.basename(pdb_file).split('.')[0] # get id from file name

    sequences = {
        "SEQRES": {},
    }

    print("\n--- extract sequence from SEQRES records ---")
    try:
        with open(pdb_file, 'r') as pdb_file:
            for record in PdbIO.PdbSeqresIterator(pdb_file):
                chain_id = record.id.split(':')[1]
                if chain_id == chain_id:
                    print(f"Chain {chain_id}: {record.seq}")
                    sequences["SEQRES"][chain_id] = str(record.seq)
    except Exception as e:
        print(f"Error: failed to parse sequence from SEQRES records: {e}")


    return sequences['SEQRES'][chain_id]

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
