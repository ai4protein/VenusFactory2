import random
from torch.utils.data import Sampler

class BatchSampler(Sampler):
    '''
    A `torch.utils.data.Sampler` which samples batches according to a
    maximum number of graph nodes.
    
    :param node_counts: array of node counts in the dataset to sample from
    :param max_batch_nodes: the maximum number of nodes in any batch,
                      including batches of a single element
    :param shuffle: if `True`, batches in shuffled order
    '''
    def __init__(self, node_counts, max_batch_nodes=10000, shuffle=True):
        
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
            max_n_node = 0
            while idx:
                if max(self.node_counts[idx[0]], max_n_node) * (len(batch) + 1) > self.max_batch_nodes:
                    break
                next_idx, idx = idx[0], idx[1:]
                current_n_node = self.node_counts[next_idx]
                if current_n_node > max_n_node:
                    max_n_node = current_n_node
                batch.append(next_idx)
            self.batches.append(batch)
    
    def __len__(self): 
        if not self.batches: self._form_batches()
        return len(self.batches)
    
    def __iter__(self):
        if not self.batches: self._form_batches()
        for batch in self.batches: yield batch
