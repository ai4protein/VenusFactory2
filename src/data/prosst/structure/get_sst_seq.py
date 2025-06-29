import torch
import os
import joblib
import warnings
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_scatter import scatter_mean, scatter_sum, scatter_max
from .encoder.gvp import AutoGraphEncoder
from .utils.data_utils import convert_graph, BatchSampler, extract_seq_from_pdb
from .build_graph import generate_graph
from .build_subgraph import generate_pos_subgraph
from pathos.multiprocessing import Pool
from pathos.threading import ThreadPool
from pathlib import Path

def iter_parallel_map(func, data, workers: int = 2):
    pool = Pool(workers)
    return pool.imap(func, data)

def iter_threading_map(func, data, workers: int = 2):
    pool = ThreadPool(workers)
    return pool.imap(func, data)

def threading_map(func, data, workers: int = 2):
    pool = ThreadPool(workers)
    return pool.map(func, data)

warnings.filterwarnings("ignore")


def predict_sturcture(model, cluster_models, dataloader, device):
    epoch_iterator = tqdm(dataloader)
    struc_label_dict = {}
    cluster_model_dict = {}

    for cluster_model_path in cluster_models:
        cluster_model_name = cluster_model_path.split("/")[-1].split(".")[0]
        struc_label_dict[cluster_model_name] = []
        cluster_model_dict[cluster_model_name] = joblib.load(cluster_model_path)

    with torch.no_grad():
        for batch in epoch_iterator:
            batch.to(device)
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)

            node_emebddings = model.get_embedding(h_V, batch.edge_index, h_E)
            graph_emebddings = scatter_mean(node_emebddings, batch.batch, dim=0).cpu()
            norm_graph_emebddings = F.normalize(graph_emebddings, p=2, dim=1)
            for name, cluster_model in cluster_model_dict.items():
                batch_structure_labels = cluster_model.predict(
                    norm_graph_emebddings
                ).tolist()
                struc_label_dict[name].extend(batch_structure_labels)

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


def subgraph_conventer(subgraph_dir, pdb_dir, max_batch_nodes, num_processes=12):
    print("---------- Load Subgraphs ----------")
    results, node_counts = [], []
    assert pdb_dir is not None, "pdb_dir is required"
    subgraph_files = sorted(
        [os.path.join(subgraph_dir, p) for p in os.listdir(subgraph_dir)]
    )

    def process_subgraph_file(subgraph_file):
        result_dict = {}
        name = subgraph_file.split("/")[-1].split(".")[0]
        result_dict["name"] = name + ".pdb"
        aa_seq = extract_seq_from_pdb(os.path.join(pdb_dir, f"{name}.pdb"))
        result_dict["aa_seq"] = aa_seq
        return result_dict, len(aa_seq)

    for result in tqdm(iter_threading_map(process_subgraph_file, subgraph_files, num_processes), total = len(subgraph_files)):
        result_dict, node_count = result
        results.append(result_dict)
        node_counts.append(node_count)
    

    def collate_fn(batch):
        # TODO: speed up
        batch_graphs = []
        for d in batch:
            subgraph_dict = torch.load(d)
            batch_graphs.extend(list(subgraph_dict.values()))

        # graph has `index_map` or other redundant attributes, remove them
        prue_batch_graphs = []
        for d in batch_graphs:
            prue_batch_graphs.append(convert_graph(d))

        batch_graphs = Batch.from_data_list(prue_batch_graphs)
        batch_graphs.node_s = torch.zeros_like(batch_graphs.node_s)
        return batch_graphs

    data_loader = DataLoader(
        subgraph_files,
        num_workers=num_processes,
        batch_sampler=BatchSampler(node_counts, max_batch_nodes, shuffle=False),
        collate_fn=collate_fn,
    )

    return data_loader, results


def graph_conventer(
    graph_dir,
    subgraph_depth,
    max_distance,
    max_batch_nodes,
    num_processes=12,
    num_threads=12,
    cache_subgraph_dir=None,
):
    print("---------- Load Graphs ----------")
    graph_files = sorted([os.path.join(graph_dir, p) for p in os.listdir(graph_dir)])
    dataset, results, node_counts = [], [], []

    def process_graph_file(
        graph_file, subgraph_depth, max_distance
    ):
        result_dict, subgraph_dict = {}, {}
        result_dict["name"] = graph_file.split("/")[-1].split(".")[0] + ".pdb"
        graph = torch.load(graph_file)
        result_dict["aa_seq"] = graph.aa_seq
        anchor_nodes = list(range(0, len(graph.aa_seq), 1))

        def process_subgraph(anchor_node):
            subgraph = generate_pos_subgraph(
                graph,
                subgraph_depth,
                max_distance,
                anchor_node,
                verbose=False,
                pure_subgraph=True,
            )[anchor_node]
            subgraph = convert_graph(subgraph)
            return anchor_node, subgraph

        # results = [process_subgraph(anchor_node) for anchor_node in anchor_nodes]
        for result in tqdm(iter_threading_map(process_subgraph, anchor_nodes, num_threads), total=len(anchor_nodes)):
            anchor, subgraph = result
            subgraph_dict[anchor] = subgraph

        subgraph_dict = dict(sorted(subgraph_dict.items(), key=lambda x: x[0]))
        if cache_subgraph_dir:
            torch.save(
                subgraph_dict,
                os.path.join(cache_subgraph_dir, f"{result_dict['name']}.pt"),
            )
            return [], result_dict, len(graph.node_s)
        subgraphs = list(subgraph_dict.values())
        return subgraphs, result_dict, len(graph.node_s)

    # multi process
    def handle_grpaph_file(graph_file):
        return process_graph_file(
            graph_file, subgraph_depth, max_distance
        )
        
    for result in tqdm(iter_parallel_map(handle_grpaph_file, graph_files, num_processes), total=len(graph_files)):
        pdb_subgraphs, result_dict, node_count = result
        dataset.append(pdb_subgraphs)
        results.append(result_dict)
        node_counts.append(node_count)
    
    def collate_fn(batch):
        batch_graphs = []
        if cache_subgraph_dir:
            for d in batch:
                name = d.split("/")[-1].split(".")[0]
                graph = torch.load(os.path.join(cache_subgraph_dir, f"{name}.pt"))
                batch_graphs.extend(graph.values())
        else:
            for d in batch:
                batch_graphs.extend(d)

        batch_graphs = Batch.from_data_list(batch_graphs)
        batch_graphs.node_s = torch.zeros_like(batch_graphs.node_s)
        return batch_graphs

    data_loader = DataLoader(
        dataset,
        num_workers=num_processes,
        batch_sampler=BatchSampler(node_counts, max_batch_nodes, shuffle=False),
        collate_fn=collate_fn,
    )

    return data_loader, results


def process_pdb_file(
    pdb_file,
    subgraph_depth,
    max_distance,
    num_threads,
    cache_subgraph_dir,
):
    result_dict, subgraph_dict = {}, {}
    result_dict["name"] = pdb_file.split("/")[-1]
    # build graph, maybe lack of some atoms
    try:
        graph = generate_graph(pdb_file, max_distance)
    except Exception as e:
        result_dict["error"] = str(e)
        return None, result_dict, 0

    # multi thread for subgraph
    result_dict["aa_seq"] = graph.aa_seq
    anchor_nodes = list(range(0, len(graph.node_s), 1))

    def process_subgraph(anchor_node):
        subgraph = generate_pos_subgraph(
            graph,
            subgraph_depth,
            max_distance,
            anchor_node,
            verbose=False,
            pure_subgraph=True,
        )[anchor_node]
        subgraph = convert_graph(subgraph)
        return anchor_node, subgraph

    for anchor_node in threading_map(process_subgraph, anchor_nodes, num_threads):
        anchor, subgraph = anchor_node
        subgraph_dict[anchor] = subgraph
    subgraph_dict = dict(sorted(subgraph_dict.items(), key=lambda x: x[0]))

    # cache graph
    if cache_subgraph_dir is not None:
        subgraph_file = os.path.join(
            cache_subgraph_dir, f"{result_dict['name'].split('.')[0]}.pt"
        )
        torch.save(subgraph_dict, subgraph_file)
        return subgraph_file, result_dict, len(anchor_nodes)
    subgraphs = list(subgraph_dict.values())
    return subgraphs, result_dict, len(anchor_nodes)


def pdb_conventer(
    pdb_files,
    subgraph_depth,
    max_distance,
    max_batch_nodes,
    error_file,
    num_processes=12,
    num_threads=12,
    cache_subgraph_dir=None,
):
    print("---------- Building Subgraphs ----------")
    error_proteins, error_messages = [], []

    dataset, results, node_counts = [], [], []
    # multi process

    def handle_pdf_file(pdb_file):
        return process_pdb_file(
            pdb_file,
            subgraph_depth,
            max_distance,
            num_threads,
            cache_subgraph_dir,
        )
        
    for result in tqdm(iter_parallel_map(handle_pdf_file, pdb_files, num_processes), total=len(pdb_files)):
        pdb_subgraphs, result_dict, node_count = result
        if pdb_subgraphs is None:
            error_proteins.append(result_dict["name"])
            error_messages.append(result_dict["error"])
            continue
        dataset.append(pdb_subgraphs)
        results.append(result_dict)
        node_counts.append(node_count)
        
    # save the error file
    if error_proteins:
        print(f"---------- Save Error File ----------")
        if error_file is None:
            error_file = os.path.join(os.path.dirname(pdb_files[0]), f"{os.path.basename(pdb_files[0]).split('.')[0]}_error.csv")
        os.makedirs(os.path.dirname(error_file), exist_ok=True)
        pd.DataFrame({"name": error_proteins, "error": error_messages}).to_csv(
            error_file, index=False
        )

    def collate_fn(batch):
        batch_graphs = []
        if cache_subgraph_dir is not None:
            for d in batch:
                name = d.split("/")[-1].split(".")[0]
                graph = torch.load(os.path.join(cache_subgraph_dir, f"{name}.pt"))
                batch_graphs.extend(graph.values())
        else:
            for d in batch:
                batch_graphs.extend(d)

        batch_graphs = Batch.from_data_list(batch_graphs)
        batch_graphs.node_s = torch.zeros_like(batch_graphs.node_s)
        return batch_graphs

    data_loader = DataLoader(
        dataset,
        num_workers=num_processes,
        batch_sampler=BatchSampler(
            node_counts, max_batch_nodes=max_batch_nodes, shuffle=False
        ),
        collate_fn=collate_fn,
    )

    return data_loader, results


class SSTPredictor:
    def __init__(
        self,
        model_path=None,
        cluster_dir=None,
        cluster_model=None,
        max_distance=10,
        subgraph_depth=None,
        max_batch_nodes=10000,
        num_processes=12,
        num_threads=16,
        device=None,
        structure_vocab_size=2048,
    ) -> None:
        """Initialize the SST predictor.
        
        Args:
            model_path: Path to the model checkpoint, defaults to static/AE.pt
            cluster_dir: Directory containing cluster models, defaults to static/
            cluster_model: List of cluster model names, defaults to ["{structure_vocab_size}.joblib"]
            max_distance: Maximum distance for edges
            subgraph_depth: Depth of subgraphs
            max_batch_nodes: Maximum number of nodes in a batch
            num_processes: Number of processes for data loading
            num_threads: Number of threads for data loading
            device: Device to run on (cuda or cpu)
            structure_vocab_size: Size of structure vocabulary (20, 64, 128, 512, 1024, 2048, 4096)
        """
        assert structure_vocab_size in [20, 64, 128, 512, 1024, 2048, 4096]
        
        if model_path is None:
            self.model_path = str(Path(__file__).parent / "static" / "AE.pt")
        else:
            self.model_path = model_path
            
        if cluster_dir is None:
            self.cluster_dir = str(Path(__file__).parent / "static")
            self.cluster_model = [f"{structure_vocab_size}.joblib"]
        else:
            self.cluster_dir = cluster_dir
            self.cluster_model = cluster_model if cluster_model is not None else [f"{structure_vocab_size}.joblib"]
            
        self.max_distance = max_distance
        self.subgraph_depth = subgraph_depth
        self.max_batch_nodes = max_batch_nodes
        self.num_processes = num_processes
        self.num_threads = num_threads
        self.structure_vocab_size = structure_vocab_size
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"---------- Load Model on {self.device} ----------")
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
        if self.device == "cpu":
            model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(self.model_path))
            model.to(self.device)
        model.eval()
        self.model = model
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"MODEL: {params:.2f}M parameters")
        
        self.cluster_models = [os.path.join(self.cluster_dir, m) for m in self.cluster_model]

    def predict_from_pdb(self, pdb_files, error_file=None, cache_subgraph_dir=None):
        """Predict structure from PDB files.
        
        Args:
            pdb_files: Single PDB file path or list of PDB file paths
            error_file: Path to save error log
            cache_subgraph_dir: Directory to cache subgraphs
            
        Returns:
            List of dictionaries containing predictions for each PDB
        """
        if isinstance(pdb_files, str):
            pdb_files = [pdb_files]
            
        data_loader, results = pdb_conventer(
            pdb_files, 
            self.subgraph_depth, 
            self.max_distance, 
            self.max_batch_nodes, 
            error_file, 
            self.num_processes, 
            self.num_threads, 
            cache_subgraph_dir
        )
        
        structures = predict_sturcture(self.model, self.cluster_models, data_loader, self.device)
        
        start, end = 0, 0
        for result in results:
            end += len(result["aa_seq"])
            for cluster_name, structure_labels in structures.items():
                result[f"{cluster_name}_sst_seq"] = structure_labels[start:end]
            start = end
            
        return results

    def predict_from_graph(self, graph_dir, cache_subgraph_dir=None):
        """Predict structure from pre-built graph files.
        
        Args:
            graph_dir: Directory containing graph files
            cache_subgraph_dir: Directory to cache subgraphs
            
        Returns:
            List of dictionaries containing predictions for each graph
        """
        data_loader, results = graph_conventer(
            graph_dir, 
            self.subgraph_depth, 
            self.max_distance, 
            self.max_batch_nodes, 
            self.num_processes, 
            self.num_threads, 
            cache_subgraph_dir
        )
        
        structures = predict_sturcture(self.model, self.cluster_models, data_loader, self.device)
        
        start, end = 0, 0
        for result in results:
            end += len(result["aa_seq"])
            for cluster_name, structure_labels in structures.items():
                result[f"{cluster_name}_s_seq"] = structure_labels[start:end]
            start = end
            
        return results

    def predict_from_subgraph(self, subgraph_dir, pdb_dir):
        """Predict structure from pre-built subgraph files.
        
        Args:
            subgraph_dir: Directory containing subgraph files
            pdb_dir: Directory containing corresponding PDB files
            
        Returns:
            List of dictionaries containing predictions for each subgraph
        """
        data_loader, results = subgraph_conventer(
            subgraph_dir,
            pdb_dir,
            self.max_batch_nodes,
            self.num_processes
        )
        
        structures = predict_sturcture(self.model, self.cluster_models, data_loader, self.device)
        
        start, end = 0, 0
        for result in results:
            end += len(result["aa_seq"])
            for cluster_name, structure_labels in structures.items():
                result[f"{cluster_name}_s_seq"] = structure_labels[start:end]
            start = end
            
        return results

