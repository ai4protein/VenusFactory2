import sys
import time
import requests
import torch
import numpy as np
import pickle
import glob
import os
import io
import logging
import tarfile
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from pathlib import Path
from typing import Any, Dict, List, Tuple, Generator
from transformers import T5Tokenizer, T5EncoderModel
from Bio.SeqUtils import seq1
import subprocess
import shutil

# Constants
SEQ_THRESHOLD = 1000  # Maximum sequence length threshold
FOLDSEEK_API_URL = "https://search.foldseek.com/api"


def extract_sequence_from_pdb(pdb_file_path: str) -> List[Tuple[str, str]]:
    from Bio import SeqIO
    from Bio.PDB import PDBParser
    
    sequences = []
    
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein', pdb_file_path)
        for model in structure:
            for chain in model:
                residues = []
                for residue in chain:
                    if residue.id[0] == ' ':
                        try:
                            residues.append(seq1(residue.resname))
                        except:
                            pass
                
                if residues:
                    seq = ''.join(residues)
                    header = f"input_pdb_chain_{chain.id}"
                    sequences.append((header, seq))
        
        return sequences
        
    except Exception as e:
        print(f"Failed to extract sequence from PDB: {e}")
        return []


# ==================== Foldseek Related Functions ====================


def download_foldseek_m8(pdb_file_path: str, output_dir: Path) -> List[str]:
    """
    Download Foldseek alignment results for a PDB file.
    
    Args:
        pdb_file_path: Path to input PDB file
        output_dir: Directory to save output files
        
    Returns:
        List of downloaded file paths
    """
    try:
        with open(pdb_file_path, "r") as f:
            pdb_content = f.read()
    except FileNotFoundError:
        raise Exception(f"Error: PDB file not found at {pdb_file_path}")
    
    databases = ["afdb-proteome", "afdb-swissprot", "afdb50", "cath50", 
                 "gmgcl_id", "mgnify_esm30", "pdb100"]
    data = {
        "q": pdb_content,
        "database[]": databases,
        "mode": "3diaa"
    }
    
    submit_response = requests.post(f"{FOLDSEEK_API_URL}/ticket", data=data)
    if submit_response.status_code != 200:
        raise Exception(f"Error submitting job: {submit_response.text}")
    
    ticket = submit_response.json()
    job_id = ticket["id"]
    print(f"Job submitted successfully. Job ID: {job_id}")
    
    status = ""
    while status != "COMPLETE":
        status_response = requests.get(f"{FOLDSEEK_API_URL}/ticket/{job_id}")
        if status_response.status_code != 200:
            raise Exception(f"Error checking status for job {job_id}: {status_response.text}")
        
        status = status_response.json()["status"]
        print(f"Current job status: {status}")
        
        if status == "ERROR":
            raise Exception(f"Job {job_id} failed. Please check the input PDB file.")
        if status != "COMPLETE":
            time.sleep(2)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_stem = Path(pdb_file_path).stem
    downloaded_files = []
    
    for db in databases:
        params = {"type": "aln", "db": db}
        download_url = f"{FOLDSEEK_API_URL}/result/download/{job_id}"
        download_response = requests.get(download_url, params=params)
        
        if download_response.status_code != 200:
            print(f"⚠️ Warning: Failed to download {db}: {download_response.text}")
            continue
        
        try:
            with tarfile.open(fileobj=io.BytesIO(download_response.content), mode='r:gz') as tar:
                for member in tar.getmembers():
                    if member.name.endswith('.m8') and '_report' not in member.name:
                        file_content = tar.extractfile(member).read()
                        
                        output_m8_path = output_dir / f"alis_{db}.m8"
                        
                        with open(output_m8_path, "wb") as f:
                            f.write(file_content)
                        
                        print(f"✅ Saved: {output_m8_path}")
                        downloaded_files.append(str(output_m8_path))
                        break
        
        except tarfile.ReadError:
            try:
                decompressed_content = gzip.decompress(download_response.content)
                output_m8_path = output_dir / f"alis_{db}.m8"
                
                with open(output_m8_path, "wb") as f:
                    f.write(decompressed_content)
                
                print(f"✅ Saved: {output_m8_path}")
                downloaded_files.append(str(output_m8_path))
            
            except gzip.BadGzipFile:
                output_m8_path = output_dir / f"alis_{db}.m8"
                
                with open(output_m8_path, "wb") as f:
                    f.write(download_response.content)
                
                print(f"✅ Saved: {output_m8_path}")
                downloaded_files.append(str(output_m8_path))
    
    print(f"\nAll downloads completed. Total files: {len(downloaded_files)}")
    return downloaded_files
# ==================== Alignment Parsing ====================

class FoldSeekAlignment:
    def __init__(self, line):
        fields = line.strip().split('\t')
        
        if len(fields) == 21:
            # for most databases
            self.qseqid = fields[0]
            self.tseqid = fields[1]
            self.pident = float(fields[2])
            self.alnlen = int(fields[3])
            self.mismatch = int(fields[4])
            self.gapopen = int(fields[5])
            self.qstart = int(fields[6])
            self.qend = int(fields[7])
            self.tstart = int(fields[8])
            self.tend = int(fields[9])
            self.evalue = float(fields[10])
            self.bitscore = float(fields[11])
            self.prob = int(fields[12])
            self.qlen = int(fields[13])
            self.tlen = int(fields[14])
            self.qaln = fields[15]
            self.taln = fields[16]
            self.tca = [float(x) for x in fields[17].split(',')]
            self.tseq = fields[18]
            self.ttaxid = fields[19]
            self.ttaxname = fields[20]
        elif len(fields) == 19:
            # for GMGC and MGnify
            self.qseqid = fields[0]
            self.tseqid = fields[1]
            self.pident = float(fields[2])
            self.alnlen = int(fields[3])
            self.mismatch = int(fields[4])
            self.gapopen = int(fields[5])
            self.qstart = int(fields[6])
            self.qend = int(fields[7])
            self.tstart = int(fields[8])
            self.tend = int(fields[9])
            self.evalue = float(fields[10])
            self.bitscore = float(fields[11])
            self.prob = int(fields[12])
            self.qlen = int(fields[13])
            self.tlen = int(fields[14])
            self.qaln = fields[15]
            self.taln = fields[16]
            self.tca = [float(x) for x in fields[17].split(',')]
            self.tseq = fields[18]
            self.ttaxid = None
            self.ttaxname = None
        else:
            raise ValueError("Invalid FoldSeek .m8 line format")

    def __str__(self):
        return f"Query: {self.query_id}, target: {self.pident}, E-value: {self.evalue}, Bit Score: {self.bit_score}"
    
    def print_alignment(self):
        print(f"Query: {self.qaln}")
        print(f"Targt: {self.taln}")

class FoldSeekAlignmentParser:
    def __init__(self, filename):
        self.filename = filename

    def parse(self):
        with open(self.filename, 'r') as f:
            alignments = [FoldSeekAlignment(line) for line in f.readlines()]
        return alignments


# ==================== Sequence Processing ====================

def check_reasonability(seq: str) -> bool:
    """Check if sequence contains only valid amino acids."""
    alphabet = set(seq)
    return alphabet.issubset(set("ACDEFGHIKLMNPQRSTVWY"))


def parse_fasta(filename: str) -> List[Tuple[str, str]]:
    """
    Parse FASTA file and return list of (header, sequence) tuples.
    
    Args:
        filename: Path to FASTA file
        
    Returns:
        List of (header, sequence) tuples
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"FASTA file not found: {filename}")
        
    with open(filename, 'r') as f:
        contents = f.read().split('>')[1:]
        data = []
        for entry in contents:
            lines = entry.split('\n')
            header = lines[0]
            sequence = ''.join(lines[1:])
            sequence = sequence.replace("*", "").replace(" ", "").upper()
            if len(sequence) <= SEQ_THRESHOLD and check_reasonability(sequence):
                data.append((header, sequence))
    return data


def parse_fasta_dict(filename: str) -> Dict[str, str]:
    """
    Parse FASTA file and return dictionary of {header: sequence}.
    
    Args:
        filename: Path to FASTA file
        
    Returns:
        Dictionary mapping headers to sequences
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"FASTA file not found: {filename}")
        
    with open(filename, 'r') as f:
        contents = f.read().split('>')[1:]
        data = {}
        for entry in contents:
            lines = entry.split('\n')
            header = lines[0]
            sequence = ''.join(lines[1:])
            sequence = sequence.replace("*", "").replace(" ", "").upper()
            data[header] = sequence
    return data


def write_fasta(data: Dict[str, str], output_file: str) -> None:
    """
    Write sequences to FASTA file.
    
    Args:
        data: Dictionary mapping headers to sequences
        output_file: Output file path
    """
    with open(output_file, 'w') as f:
        for header, sequence in data.items():
            f.write(f">{header}\n")
            f.write(f"{sequence}\n")


# ==================== MMseqs Related Functions ====================

def read_tsv(file_name: str) -> Tuple[List[str], List[List[str]]]:
    """
    Read TSV file from MMseqs2 output.
    
    Args:
        file_name: Path to TSV file
        
    Returns:
        Tuple of (header, data rows)
    """
    seq_info_list = []
    header = ["query", "target", "pident", "fident", "nident", "alnlen", "bits", "tseq", "evalue"]
    
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"TSV file not found: {file_name}")
    
    with open(file_name, 'r') as f:
        for line in f:
            if line.strip():
                seq_info = line.strip().split('\t')
                seq_info_list.append(seq_info)

    if seq_info_list and len(seq_info_list[0]) != len(header):
        print(f"Warning: Expected {len(header)} columns, got {len(seq_info_list[0])}")
        
    return header, seq_info_list


def write_fasta_from_tsv(header: List[str], seq_info_list: List[List[str]], 
                         out_file_name: str, evalue: float = None) -> None:
    """
    Output FASTA file from TSV data.
    
    Args:
        header: Column names
        seq_info_list: Data rows
        out_file_name: Output file path
        evalue: E-value threshold (None for no filtering)
    """
    tag_index = header.index("target")
    seq_index = header.index("tseq")
    evalue_index = header.index("evalue")

    with open(out_file_name, 'w') as f:
        for seq_info in seq_info_list:
            if evalue is None or float(seq_info[evalue_index]) <= evalue:
                f.write(f">{seq_info[tag_index]}\n")
                f.write(f"{seq_info[seq_index]}\n")


def read_mmseqs_cluster_tsv(file_name: str) -> List[List[str]]:
    """
    Read MMseqs2 cluster TSV file.
    
    Args:
        file_name: Path to cluster TSV file
        
    Returns:
        List of [representative, member] pairs
    """
    cluster_info = []
    
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Cluster file not found: {file_name}")
    
    with open(file_name, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    cluster_info.append(parts[:2])

    return cluster_info


def read_selected_cluster_names(file_name: str) -> List[str]:
    """
    Read selected cluster names from file.
    
    Args:
        file_name: Path to file containing cluster names
        
    Returns:
        List of cluster names
    """
    cluster_names = []
    
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Cluster names file not found: {file_name}")
    
    with open(file_name, 'r') as f:
        for line in f:
            if line.strip():
                cluster_names.append(line.strip())

    return cluster_names


def run_mmseqs_search(input_fasta: str, database_path: str, output_tsv: str,
                      threads: int = 96, num_iterations: int = 2, 
                      max_seqs: int = 2500) -> bool:
    """
    Run MMseqs2 easy-search.
    
    Args:
        input_fasta: Input FASTA file
        database_path: Path to MMseqs database
        output_tsv: Output TSV file
        threads: Number of threads
        num_iterations: Number of search iterations
        max_seqs: Maximum number of sequences
        
    Returns:
        True if successful, False otherwise
    """
    tmp_dir = Path(output_tsv).parent / "mmseqs_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "mmseqs", "easy-search",
        input_fasta,
        database_path,
        output_tsv,
        str(tmp_dir),
        "--format-output", "query,target,pident,fident,nident,alnlen,bits,tseq,evalue",
        "--threads", str(threads),
        "--num-iterations", str(num_iterations),
        "--max-seqs", str(max_seqs),
        "--search-type", "1"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"MMseqs search completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"MMseqs search failed: {e.stderr}")
        return False
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def run_mmseqs_cluster(input_fasta: str, output_prefix: str, 
                       min_seq_id: float = 0.5, threads: int = 96) -> bool:
    """
    Run MMseqs2 easy-cluster.
    
    Args:
        input_fasta: Input FASTA file
        output_prefix: Output file prefix
        min_seq_id: Minimum sequence identity
        threads: Number of threads
        
    Returns:
        True if successful, False otherwise
    """
    tmp_dir = Path(output_prefix).parent / "cluster_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "mmseqs", "easy-cluster",
        input_fasta,
        output_prefix,
        str(tmp_dir),
        "--min-seq-id", str(min_seq_id),
        "--threads", str(threads)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"MMseqs clustering completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"MMseqs clustering failed: {e.stderr}")
        return False
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ==================== ProstT5 Embedding Functions ====================

def calculate_representation(model: T5EncoderModel, tokenizer: T5Tokenizer, 
                            data: List[Tuple[str, str]], logger: logging.Logger, start_time: float, 
                            batch_size: int = 8) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Calculate ProstT5 representations for sequences.
    
    Args:
        model: ProstT5 model
        tokenizer: ProstT5 tokenizer
        data: List of (header, sequence) tuples
        logger: Logger instance
        start_time: Start time for progress tracking
        batch_size: Number of sequences to process in each batch (1, 8, or 16)
        
    Returns:
        Tuple of (sequence_labels, sequence_strs, representations)
    """
    sequence_labels, sequence_strs, sequence_representations = [], [], []
    total_sequences = len(data)
    num_batches = (total_sequences + batch_size - 1) // batch_size  # Ceiling division

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_sequences)

        batch_labels = [data[i][0] for i in range(start_idx, end_idx)]
        batch_strs = [data[i][1] for i in range(start_idx, end_idx)]
        
        # Add ProstT5 prefix
        batch_strs_model = []
        for seq in batch_strs:
            if seq.isupper():
                batch_strs_model.append("<AA2fold>" + " " + " ".join(list(seq)))
            else:
                batch_strs_model.append("<fold2AA>" + " " + " ".join(list(seq)))

        # Tokenize sequences
        ids = tokenizer.batch_encode_plus(
            batch_strs_model, 
            add_special_tokens=True, 
            padding="longest",
            return_tensors='pt'
        ).to('cuda')

        # Generate embeddings
        with torch.no_grad():
            token_representations = model(
                ids.input_ids, 
                attention_mask=ids.attention_mask
            )
        
        # Extract mean representation
        for i in range(len(batch_strs)):
            seq_len = len(batch_strs[i])
            repr_mean = token_representations.last_hidden_state[i, 1:seq_len+1].mean(dim=0)
            sequence_representations.append(repr_mean.cpu().numpy())
            sequence_labels.append(batch_labels[i])
            sequence_strs.append(batch_strs[i])

        # Print progress every 10 batches or for the last batch
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - start_time
            progress = 100.0 * (batch_idx + 1) / num_batches
            sequences_processed = min(end_idx, total_sequences)
            logger.info(f"Batch {batch_idx + 1}/{num_batches} processed ({sequences_processed}/{total_sequences} sequences). Time: {elapsed:.2f}s, Progress: {progress:.2f}%")

        del batch_labels, batch_strs, batch_strs_model, token_representations
        torch.cuda.empty_cache()

    return sequence_labels, sequence_strs, np.vstack(sequence_representations)


def save_representation(sequence_labels: List[str], sequence_strs: List[str], 
                       representation: np.ndarray, output_file: str) -> None:
    """
    Save representations to pickle file.
    
    Args:
        sequence_labels: Sequence labels
        sequence_strs: Sequence strings
        representation: Numpy array of representations
        output_file: Output pickle file path
    """
    with open(output_file, 'wb') as f:
        pickle.dump({
            "labels": sequence_labels, 
            "sequences": sequence_strs, 
            "representations": representation
        }, f)
    print(f"Saved representations to {output_file}")


# ==================== Tree Building Functions ====================

def distance_to_reference(repr: np.ndarray, ref: np.ndarray) -> float:
    return np.min(np.linalg.norm(repr - ref, axis=1))


def smoothsegment(seg: np.ndarray, Nsmooth: int = 100) -> np.ndarray:
    return np.concatenate([[seg[0]], np.linspace(seg[1], seg[2], Nsmooth), [seg[3]]])


def build_and_visualize_tree(refseq_pkl: str, discovered_pkl: str, ec_pkl: str, ec_csv: str,
                            top_n_threshold: int, output_dir: Path) -> Tuple[str, str]:
    """
    Build hierarchical clustering tree and create visualization.
    
    Args:
        refseq_pkl: Path to reference sequences pickle
        discovered_pkl: Path to discovered sequences pickle
        ec_pkl: Path to ec sequence pickle
        ec_csv: Path to ec sequence information
        top_n_threshold: Number of top sequences to include
        output_dir: Output directory for plots
        
    Returns:
        Tuple of (plot_path, label_file_path)
    """
    mpl.rcParams['font.size'] = 12
    # mpl.rcParams['font.family'] = 'Arial'
    
    # Load data
    result_refseq = pickle.load(open(refseq_pkl, "rb"))
    result_discover = pickle.load(open(discovered_pkl, "rb"))
    result_ec = pickle.load(open(ec_pkl, "rb"))
    
    repr_refseq = result_refseq["representations"]
    repr_discover = result_discover["representations"]
    repr_ec = result_ec["representations"]

    label_refseq = result_refseq["labels"]
    label_discover = result_discover["labels"]
    label_ec = result_ec["labels"]

    sequence_refseq = result_refseq["sequences"]
    sequence_discover = result_discover["sequences"]
    sequence_ec = result_ec["sequences"]

    data_ec = pd.read_csv(ec_csv)
    ref = repr_refseq
    
    # Calculate distances
    distance_to_ref_discover = []
    full_length_discover = repr_discover.shape[0]
    for i in range(full_length_discover):
        distance_to_ref_discover.append(distance_to_reference(repr_discover[i], ref))

    # EC
    distance_to_ref_ec = []
    full_length_ec = repr_ec.shape[0]
    for i in range(full_length_ec):
        distance_to_ref_ec.append(distance_to_reference(repr_ec[i], ref))
    
    # Use same threshold for EC sequences
    top_n_threshold_ec = min(top_n_threshold, full_length_ec)
    
    # Merge representations
    merge_repr_discover = np.concatenate([repr_refseq, repr_discover[np.argsort(distance_to_ref_discover)[0:top_n_threshold]], repr_ec[np.argsort(distance_to_ref_ec)[0:top_n_threshold_ec]]])
    merge_label_discover = np.concatenate([label_refseq, np.array(label_discover)[np.argsort(distance_to_ref_discover)[0:top_n_threshold]], np.array(label_ec)[np.argsort(distance_to_ref_ec)[0:top_n_threshold_ec]]])
    
    # Build dendrogram
    distance_matrix_discover = pdist(merge_repr_discover, metric='euclidean')
    linkage_matrix_discover = linkage(distance_matrix_discover, method='ward')
    hierarchal_cluster = fcluster(linkage_matrix_discover, 4, criterion='distance')

    # Create dendrogram
    plt.figure(figsize=(12, 3))
    tree_dict_discover = dendrogram(linkage_matrix_discover, color_threshold=4, no_labels=True)
    plt.close()

    # Extract coordinates
    icoord = np.array(tree_dict_discover['icoord'])
    dcoord = np.array(tree_dict_discover['dcoord'])
    color_list = tree_dict_discover['color_list']
    color_collection = list(set(color_list))

    # Transform coordinates
    dcoord = -np.log(dcoord + 1)
    gap = 0.0
    imax, imin = icoord.max(), icoord.min()
    icoord = ((icoord - imin) / (imax - imin) * (1 - gap) + gap / 2) * 2 * np.pi

    # Create radial plot
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True, rlabel_position=0)

    for i in range(len(icoord)):
        xs = smoothsegment(icoord[i])
        ys = smoothsegment(dcoord[i])
        color_value = (color_collection.index(color_list[i])) / max(len(color_collection) - 1, 1)
        plt.plot(xs, ys, color=plt.cm.tab10_r(color_value), linewidth=1, zorder=0)
    # Add leaves
    index2rad = np.pi * 2 / len(tree_dict_discover["leaves"])
    leave_index_list = tree_dict_discover["leaves"]
    
    for i in range(len(leave_index_list)):
        leave_index = leave_index_list[i]
        rad = i*index2rad
        if leave_index < repr_refseq.shape[0]:        
            plt.plot([rad, rad], 
                    [0.1, 0.2], 
                    color="tab:green", linewidth=1, zorder=5)
        elif leave_index >= repr_refseq.shape[0] + top_n_threshold:  # from EC
            target_ec = "0.0.0.0"
            contain_target_ec = target_ec in data_ec[data_ec["UniProt_ID"]==merge_label_discover[leave_index]]["EC"].values

            if contain_target_ec:
                plt.plot([rad, rad], 
                        [0.2, 0.3], 
                        color="tab:green", linewidth=1, zorder=5)
            else:
                plt.plot([rad, rad], 
                        [0.2, 0.3], 
                        color="tab:grey", linewidth=1, zorder=5)
        else:
            pass

        plt.plot([rad, rad], 
                [0.0, 0.1], 
                color=plt.cm.Spectral((hierarchal_cluster[leave_index]-1)%10/9), linewidth=1, zorder=5)

    plt.ylim(-3.5, 0.4)
    plt.grid(False)
    plt.xticks(np.linspace(0, 2*np.pi, 60), [" "]+["%d"%x for x in np.linspace(0, len(leave_index_list), 60)][1:])
    plt.yticks([])
    plt.tight_layout()
    
    plt.gca().spines['polar'].set_visible(False)

    # Save plot
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "dendrogram.png"
    plt.savefig(plot_path, dpi=300, transparent=False)
    plt.close()

    # Save labels
    label_file = output_dir / "dendrogram_labels.tsv"
    with open(label_file, 'w') as f:
        f.write("index\ttype\tlabel\tcluster\n")
        for i, leave_idx in enumerate(leave_index_list):
            seq_type = "refseq" if leave_idx < repr_refseq.shape[0] else "discovered"
            f.write(f"{i}\t{seq_type}\t{merge_label_discover[leave_idx]}\t{hierarchal_cluster[leave_idx]}\n")

    return str(plot_path), str(label_file)