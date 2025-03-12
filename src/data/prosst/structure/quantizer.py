import biotite
import joblib
import math
import numpy as np
import os
import scipy.spatial as spa
import torch
import torch.nn.functional as F
from Bio import PDB
from Bio.SeqUtils import seq1
from pathlib import Path
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from tqdm import tqdm
from typing import List
from biotite.sequence import ProteinSequence
from biotite.structure import filter_backbone, get_chains
from biotite.structure.io import pdb, pdbx
from biotite.structure.residues import get_residues
from .encoder import AutoGraphEncoder


def _normalize(tensor, dim=-1):
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
    )


def _rbf(D, D_min=0.0, D_max=20.0, D_count=16, device="cpu"):
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
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


def _positional_embeddings(edge_index, num_embeddings=16, period_range=[2, 1000]):
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
        node_level: residue or secondary_structure
        node_s_type: ss3, ss8, foldseek
        max_distance: cut off
        foldseek_fasta_file: foldseek fasta file path
        foldseek_fasta_multi_chain: pdb multi chain for foldseek fasta

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
    one_letter_seq = "".join([seq1(aa) for aa in seq])

    # aa means amino acid
    coords = list(zip(aa_coords["N"], aa_coords["CA"], aa_coords["C"], aa_coords["O"]))
    coords = torch.tensor(coords)
    # mask out the missing coordinates
    mask = torch.isfinite(coords.sum(dim=(1, 2)))
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
    node_s, node_v, edge_s, edge_v = map(
        torch.nan_to_num, (node_s, node_v, edge_s, edge_v)
    )
    data = Data(
        node_s=node_s,
        node_v=node_v,
        edge_index=edge_index,
        edge_s=edge_s,
        edge_v=edge_v,
        distances=distances,
        aa_seq=one_letter_seq,
    )

    return data


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
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
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
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return seq


def generate_pos_subgraph(
    graph_data,
    subgraph_depth=None,
    subgraph_interval=1,
    max_distance=10,
    anchor_nodes=None,
    pure_subgraph=False,
    device="cuda" if torch.cuda.is_available() else "cpu"
):

    # move graph_data to GPU
    graph_data = Data(
        node_s=graph_data.node_s.to(device) if torch.is_tensor(graph_data.node_s) else torch.tensor(graph_data.node_s, device=device),
        node_v=graph_data.node_v.to(device) if torch.is_tensor(graph_data.node_v) else torch.tensor(graph_data.node_v, device=device),
        edge_index=graph_data.edge_index.to(device) if torch.is_tensor(graph_data.edge_index) else torch.tensor(graph_data.edge_index, device=device),
        edge_s=graph_data.edge_s.to(device) if torch.is_tensor(graph_data.edge_s) else torch.tensor(graph_data.edge_s, device=device),
        edge_v=graph_data.edge_v.to(device) if torch.is_tensor(graph_data.edge_v) else torch.tensor(graph_data.edge_v, device=device),
        distances=graph_data.distances.to(device) if torch.is_tensor(graph_data.distances) else torch.tensor(graph_data.distances, device=device),
        aa_seq=graph_data.aa_seq
    )

    distances = graph_data.distances
    if subgraph_depth is None:
        subgraph_depth = 50

    # Calculate anchor nodes if not provided
    if anchor_nodes is None:
        anchor_nodes = list(range(0, len(graph_data.aa_seq), subgraph_interval))
    anchor_nodes_tensor = torch.tensor(anchor_nodes, device=device)  # Move anchor nodes to device

    # Get the k nearest neighbors for ALL anchor nodes (batched)
    k = 50
    nearest_indices = torch.argsort(distances, dim=1)[:, :k]  # (num_nodes, k)
    distance_mask = torch.gather(distances, 1, nearest_indices) < max_distance  # (num_nodes, k)
    nearest_indices = torch.where(distance_mask, nearest_indices, torch.tensor(-1, device=device))  # (num_nodes, k)

    subgraph_dict = {}

    for anchor_node in anchor_nodes: #Reverted back to for loop to ensure everything works with batches
        try:

            #Get neighbors for each anchornode
            k_neighbors = nearest_indices[anchor_node]
            k_neighbors = k_neighbors[k_neighbors != -1]

            if len(k_neighbors) == 0:  # Skip if no neighbors found
                    continue

            if len(k_neighbors) > 30:
                k_neighbors = k_neighbors[:40]

            k_neighbors, _ = torch.sort(k_neighbors)

            sub_matrix = distances.index_select(0, k_neighbors).index_select(1, k_neighbors)

            # Create edge indices efficiently
            sub_edges = torch.nonzero(sub_matrix < max_distance, as_tuple=False)
            mask = sub_edges[:, 0] != sub_edges[:, 1]
            sub_edge_index = sub_edges[mask]

            if len(sub_edge_index) == 0:  # Skip if no edges found
                continue
            # Move edge_index to GPU only when needed
            edge_index_device = graph_data.edge_index.to(device)
            original_edge_index = k_neighbors[sub_edge_index]

            # More memory efficient edge matching
            matches = []
            for edge in original_edge_index:
                match = (edge_index_device[0] == edge[0]) & (edge_index_device[1] == edge[1])
                matches.append(match)
            matches = torch.stack(matches)
            edge_to_feature_idx = torch.nonzero(matches, as_tuple=True)[0].to(device)
            if len(edge_to_feature_idx) == 0:  # Skip if no matching edges
                continue
            #Create data
            new_node_s = graph_data.node_s[k_neighbors].to(device)
            new_node_v = graph_data.node_v[k_neighbors].to(device)
            new_edge_s = graph_data.edge_s[edge_to_feature_idx].to(device)
            new_edge_v = graph_data.edge_v[edge_to_feature_idx].to(device)

            result = Data(
                edge_index=sub_edge_index.T.to(device),
                edge_s=new_edge_s.to(device),
                edge_v=new_edge_v.to(device),
                node_s=new_node_s.to(device),
                node_v=new_node_v.to(device),
            )
            if not pure_subgraph:
                result.index_map = {
                int(old_id.to(device).item()): new_id
                for new_id, old_id in enumerate(k_neighbors)
                }
            subgraph_dict[anchor_node] = result
        except Exception as e:
             print(f"Error processing anchor node {anchor_node}: {str(e)}")
             continue

    return subgraph_dict

def load_structure(fpath, chain=None):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    if fpath.endswith("cif"):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith("pdb"):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    all_chains = get_chains(structure)
    if len(all_chains) == 0:
        raise ValueError("No chains found in the input file.")
    if chain is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f"Chain {chain} not found in input file")
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
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
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
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
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

def predict_structure(model, cluster_models, dataloader, datalabels, device):
    epoch_iterator = dataloader
    struc_label_dict = {}
    cluster_model_dict = {}

    for cluster_model_path in cluster_models:
        cluster_model_name = cluster_model_path.split("/")[-1].split(".")[0]
        struc_label_dict[cluster_model_name] = {}
        cluster_model_dict[cluster_model_name] = joblib.load(cluster_model_path)

    with torch.no_grad():
        for batch, label_dict in zip(epoch_iterator, datalabels):
            batch.to(device)
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)

            node_emebddings = model.get_embedding(h_V, batch.edge_index, h_E)
            graph_emebddings = scatter_mean(node_emebddings, batch.batch, dim=0).to(device)
            norm_graph_emebddings = F.normalize(graph_emebddings, p=2, dim=1)
            struc_label_dict[cluster_model_name][label_dict['name']]={}
            for name, cluster_model in cluster_model_dict.items():
                batch_structure_labels = cluster_model.predict(
                    norm_graph_emebddings.cpu()
                ).tolist()
                struc_label_dict[name][label_dict['name']]['seq']=label_dict['aa_seq']
                struc_label_dict[name][label_dict['name']]['struct']=batch_structure_labels

    return struc_label_dict


def get_embeds(model, dataloader, device, pooling="mean"):
    epoch_iterator = tqdm(dataloader)
    embeds = []
    with torch.no_grad():
        for batch in epoch_iterator:
            batch.to(device)
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            node_embeds = model.get_embedding(h_V, batch.edge_index, h_E).cpu()
            if pooling == "mean":
                graph_embeds = scatter_mean(node_embeds, batch.batch.cpu(), dim=0)
            elif pooling == "sum":
                graph_embeds = scatter_sum(node_embeds, batch.batch.cpu(), dim=0)
            elif pooling == "max":
                graph_embeds, _ = scatter_max(node_embeds, batch.batch.cpu(), dim=0)
            else:
                raise ValueError("pooling should be mean, sum or max")
            embeds.append(graph_embeds)

    embeds = torch.cat(embeds, dim=0)
    norm_embeds = F.normalize(embeds, p=2, dim=1)
    return norm_embeds

def process_pdb_file(
    pdb_file,
    subgraph_depth,
    subgraph_interval,
    max_distance,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    result_dict, subgraph_dict = {}, {}
    result_dict["name"] = Path(pdb_file).name
    
    try:
        graph = generate_graph(pdb_file, max_distance)
    except Exception as e:
        print(f"Error in processing {pdb_file}")
        result_dict["error"] = str(e)
        return None, result_dict, 0

    result_dict["aa_seq"] = graph.aa_seq
    anchor_nodes = list(range(0, len(graph.node_s), subgraph_interval)) #Define anchor nodes
    try: #Run subgraph generation
        subgraph_dict = generate_pos_subgraph(
            graph,
            subgraph_depth,
            subgraph_interval,
            max_distance,
            anchor_nodes=anchor_nodes,
            pure_subgraph=True,
            device=device
        )
        #Move all subgraphs to GPU
        for key in subgraph_dict.keys():
            subgraph_dict[key] = convert_graph(subgraph_dict[key])
    except Exception as e:
        print(f"Error processing subgraph {e}")
        return None, result_dict, 0


    subgraph_dict = dict(sorted(subgraph_dict.items(), key=lambda x: x[0]))
    subgraphs = list(subgraph_dict.values())
    return subgraphs, result_dict, len(anchor_nodes)


def pdb_converter(
    pdb_files,
    subgraph_depth,
    subgraph_interval,
    max_distance,
    device="cuda" if torch.cuda.is_available() else "cpu",
    batch_size=32 
):
    error_proteins, error_messages = [], []
    dataset, results, node_counts = [], [], []

    for i in tqdm(range(0, len(pdb_files), batch_size), desc="Processing PDB files"):
        batch = pdb_files[i:i + batch_size]

        for pdb_file in batch:
            pdb_subgraphs, result_dict, node_count = process_pdb_file(
                pdb_file,
                subgraph_depth,
                subgraph_interval,
                max_distance,
                device=device
            )

            if pdb_subgraphs is None:
                error_proteins.append(result_dict["name"])
                error_messages.append(result_dict["error"])
                continue
            dataset.append(pdb_subgraphs)
            results.append(result_dict)
            node_counts.append(node_count)

    if error_proteins:
        print(f"Found {len(error_proteins)} errors:")
        for name, msg in zip(error_proteins, error_messages):
            print(f"{name}: {msg}")

    def collate_fn(batch):
        batch_graphs = []
        for d in batch:
            batch_graphs.extend(d)
        batch_graphs = Batch.from_data_list(batch_graphs)
        batch_graphs.node_s = torch.zeros_like(batch_graphs.node_s)
        return batch_graphs

    def data_loader():
        for item in dataset:
            yield collate_fn([item])

    return data_loader(), results


class PdbQuantizer:

    def __init__(
        self,
        structure_vocab_size=2048,
        max_distance=10,
        subgraph_depth=None,
        subgraph_interval=1,
        anchor_nodes=None,
        model_path=None,
        cluster_dir=None,
        cluster_model=None,
        device=None,
        batch_size=16,
    ) -> None:
        assert structure_vocab_size in [20, 64, 128, 512, 1024, 2048, 4096]
        self.batch_size = batch_size
        self.max_distance = max_distance
        self.subgraph_depth = subgraph_depth
        self.subgraph_interval = subgraph_interval
        self.anchor_nodes = anchor_nodes
        if model_path is None:
            self.model_path = str(Path(__file__).parent / "static" / "AE.pt")
        else:
            self.model_path = model_path
        self.structure_vocab_size = structure_vocab_size

        if cluster_dir is None:
            self.cluster_dir = str(Path(__file__).parent / "static")
            self.cluster_model = [
                Path(self.cluster_dir) / f"{structure_vocab_size}.joblib",
            ]
        else:
            self.cluster_dir = cluster_dir
            self.cluster_model = cluster_model

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        node_dim = (256, 32)
        edge_dim = (64, 2)
        model = AutoGraphEncoder(
            node_in_dim=(20, 3),
            node_h_dim=node_dim,
            edge_in_dim=(32, 1),
            edge_h_dim=edge_dim,
            num_layers=6,
        )
        model.load_state_dict(torch.load(self.model_path))
        model = model.to(self.device)
        model = model.eval()
        self.model = model
        self.cluster_models = [
            os.path.join(self.cluster_dir, m) for m in self.cluster_model
        ]

    def __call__(self, pdb_files, return_residue_seq=False):
        if isinstance(pdb_files, str):
            pdb_files = [pdb_files]
        elif isinstance(pdb_files, list):
            pass
        else:
            raise ValueError("pdb_files should be either a string or a list of strings")
        data_loader, results = pdb_converter(
            pdb_files,
            self.subgraph_depth,
            self.subgraph_interval,
            self.max_distance,
            device=self.device,
            batch_size=self.batch_size
        )
        structures = predict_structure(
            self.model, self.cluster_models, data_loader, results, self.device
        )
        if not return_residue_seq:
            for clusterModelLabels in structures.keys():
                for structureDict in structures[clusterModelLabels].keys():
                    structures[clusterModelLabels][structureDict].pop('seq', None)
        return structures

