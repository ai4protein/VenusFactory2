import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data

def generate_pos_subgraph(graph_data, subgraph_depth=None, 
                          max_distance=10, anchor_nodes=None, verbose=False, pure_subgraph=False):
    """
    generate subgraphs from graph data
    parmas:
        graph_data: pytorch geometric data
        subgraph_depth: knn: k
        max_distance: cut off distance
        anchor_nodes: anchor nodes
        verbose: print progress bar
        pure_subgraph: only return subgraph, no other information
    
    return:
        subgraph_dict: {center_node: subgraph_data, ...}
    """
    distances = graph_data.distances
    subgraph_dict = {}
    if subgraph_depth is None:
        subgraph_depth = 50
    sorted_indices = np.argsort(distances, axis=1)[:, :50]
    mask = distances[np.arange(distances.shape[0])[:, None], sorted_indices] < 10
    nearest_indices = np.where(mask, sorted_indices, -1)

    
    def quick_get_anchor_graph(anchor_node):
        k_neighbors_indices = nearest_indices[anchor_node][nearest_indices[anchor_node] != -1]
        k_neighbors_indices = k_neighbors_indices[:40]
        # reorder the indices
        k_neighbors_indices = np.array(sorted(k_neighbors_indices.tolist()))
        sub_matrix = distances[k_neighbors_indices][:, k_neighbors_indices]
        sub_edge_index = np.transpose(np.nonzero(sub_matrix < max_distance))
        
        # remove loop
        mask = sub_edge_index[:, 0] != sub_edge_index[:, 1]
        sub_edge_index = sub_edge_index[mask]
        original_edge_index = k_neighbors_indices[sub_edge_index]
        matches = np.all(np.transpose(graph_data.edge_index.numpy())[:, None] == original_edge_index, axis=2)
        edge_to_feature_idx = np.nonzero(matches.any(axis=1))[0]
        
        new_node_s = graph_data.node_s[k_neighbors_indices]
        new_node_v = graph_data.node_v[k_neighbors_indices]
        new_edge_s = graph_data.edge_s[edge_to_feature_idx]
        new_edge_v = graph_data.edge_v[edge_to_feature_idx]
        
        if pure_subgraph:
            return Data(
                        edge_index=torch.tensor(sub_edge_index).T,
                        edge_s=new_edge_s, edge_v=new_edge_v,
                        node_s=new_node_s, node_v=new_node_v,
                    )
        else:
            # reindex the edge index
            new_index_mapping = {int(old_id): new_id for new_id, old_id in enumerate(k_neighbors_indices)}
            # print(anchor_node+1, ",".join([str(i+1) for i in new_index_mapping.keys()]))
            return Data(
                        index_map=new_index_mapping,
                        edge_index=torch.tensor(sub_edge_index).T,
                        edge_s=new_edge_s, edge_v=new_edge_v,
                        node_s=new_node_s, node_v=new_node_v,
                    )

    if anchor_nodes is not None:
        if type(anchor_nodes) == int:
            subgraph_dict[anchor_nodes] = quick_get_anchor_graph(anchor_nodes)
        elif type(anchor_nodes) == list:
            for anchor_node in anchor_nodes:
                subgraph_dict[anchor_node] = quick_get_anchor_graph(anchor_node)
            
    else:
        # loop over all nodes
        anchor_nodes = len(graph_data.aa_seq)
        if verbose:
            for anchor_node in tqdm(range(anchor_nodes)):
                subgraph_dict[anchor_node] = quick_get_anchor_graph(anchor_node)
        else:
            for anchor_node in range(anchor_nodes):
                subgraph_dict[anchor_node] = quick_get_anchor_graph(anchor_node)
            
    return subgraph_dict
