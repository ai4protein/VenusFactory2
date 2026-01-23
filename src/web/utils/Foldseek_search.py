import os
import io
import time
import json
import gzip
import tarfile
from pathlib import Path
from datetime import datetime
import requests
from .common_utils import get_save_path, sanitize_filename


SEQ_THRESHOLD = 1000
FOLDSEEK_API_URL = "https://search.foldseek.com/api"

def download_foldseek_m8(pdb_file_path: str, output_dir: Path) -> List[str]:
    """Download FoldSeek alignment results"""
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

class FoldSeekAlignment:
    """FoldSeek alignment parser"""
    def __init__(self, line):
        fields = line.strip().split('\t')
        
        if len(fields) == 21:
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


class FoldSeekAlignmentParser:
    """Parser for FoldSeek alignment files"""
    def __init__(self, filename):
        self.filename = filename

    def parse(self):
        with open(self.filename, 'r') as f:
            alignments = [FoldSeekAlignment(line) for line in f.readlines()]
        return alignments


def prepare_foldseek_sequences(
    alignments_files: List[str], 
    output_fasta: Path, 
    protect_start: int, 
    protect_end: int,
) -> int:
    """Extract sequences from FoldSeek alignments that cover the protected region"""
    alignments_dbname = [
        "afdb_proteome",
        "afdb_swissprot",
        "afdb50",
        "cath50",
        "gmgcl_id",
        "mgnify_esm30",
        "pdb100",
    ]
    count = 0
    with open(output_fasta, "w", encoding="utf-8") as f_out:
        for db_name, filename in zip(alignments_dbname, alignments_files):
            parser = FoldSeekAlignmentParser(filename)
            alignments = parser.parse()
            for alignment in alignments:
                if alignment.qstart <= protect_start and alignment.qend >= protect_end:
                    f_out.write(f">{db_name} {alignment.tseqid.split(' ')[0]}\n")
                    f_out.write(f"{alignment.tseq}\n")
                    count += 1
    return count

def get_foldseek_sequences(pdb_file_path: str, protect_start: int, protect_end: int) -> int:
    """Extract sequences from FoldSeek alignments"""
    # FoldSeek Search
    foldseek_dir = get_save_path("FoldSeek", "Download_data")
    download_files = download_foldseek_m8(str(pdb_file_path), foldseek_dir)
    
    # Extract sequences from FoldSeek alignments
    foldseek_fasta = foldseek_dir / f"{os.path.basename(pdb_file_path).replace(".pdb","")}_{datetime.now().strftime('%Y%m%d%H%M%S')}.fasta"
    total_sequences = prepare_foldseek_sequences(download_files, foldseek_fasta, protect_start, protect_end)
    return foldseek_fasta, total_sequences


if __name__ == "__main__":
    example_pdb_path = r"download/alphafold2_structures/A0A1B0GTW7.pdb"
    
    foldseek_fasta, total_sequences = get_foldseek_sequences(example_pdb_path, 1, 10)
    
    if foldseek_fasta:
        print(f"Success! Fasta Path: {foldseek_fasta}")
        print(f"Result Info: {total_sequences}")
    else:
        print("Failed!")
