"""
Download FoldSeek alignment results (.m8) by job_id; parse alignments; prepare FASTA.
"""
import io
import gzip
import tarfile
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Union

import requests

from .foldseek_submit import FOLDSEEK_API_URL, DEFAULT_DATABASES

SEQ_THRESHOLD = 1000


def download_foldseek_m8(
    job_id: str,
    output_dir: Union[Path, str],
    databases: Optional[List[str]] = None,
) -> List[str]:
    """Download alignment result files for a FoldSeek job_id. Returns list of .m8 file paths."""
    output_dir = Path(output_dir)
    databases = databases or DEFAULT_DATABASES
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_files = []

    for db in databases:
        params = {"type": "aln", "db": db}
        download_url = f"{FOLDSEEK_API_URL}/result/download/{job_id}"
        download_response = requests.get(download_url, params=params)

        if download_response.status_code != 200:
            continue

        try:
            with tarfile.open(fileobj=io.BytesIO(download_response.content), mode="r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".m8") and "_report" not in member.name:
                        file_content = tar.extractfile(member).read()
                        output_m8_path = output_dir / f"alis_{db}.m8"
                        with open(output_m8_path, "wb") as f:
                            f.write(file_content)
                        downloaded_files.append(str(output_m8_path))
                        break
        except tarfile.ReadError:
            try:
                decompressed_content = gzip.decompress(download_response.content)
                output_m8_path = output_dir / f"alis_{db}.m8"
                with open(output_m8_path, "wb") as f:
                    f.write(decompressed_content)
                downloaded_files.append(str(output_m8_path))
            except gzip.BadGzipFile:
                output_m8_path = output_dir / f"alis_{db}.m8"
                with open(output_m8_path, "wb") as f:
                    f.write(download_response.content)
                downloaded_files.append(str(output_m8_path))

    return downloaded_files


class FoldSeekAlignment:
    """Single FoldSeek alignment line (.m8 format)."""

    def __init__(self, line: str):
        fields = line.strip().split("\t")
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
            self.tca = [float(x) for x in fields[17].split(",")]
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
            self.tca = [float(x) for x in fields[17].split(",")]
            self.tseq = fields[18]
            self.ttaxid = None
            self.ttaxname = None
        else:
            raise ValueError("Invalid FoldSeek .m8 line format")


class FoldSeekAlignmentParser:
    """Parse a FoldSeek .m8 alignment file."""

    def __init__(self, filename: str):
        self.filename = filename

    def parse(self) -> List[FoldSeekAlignment]:
        with open(self.filename) as f:
            return [FoldSeekAlignment(line) for line in f.readlines()]


def prepare_foldseek_sequences(
    alignments_files: List[str],
    output_fasta: Path,
    protect_start: int,
    protect_end: int,
) -> int:
    """Extract sequences from FoldSeek alignments that cover the protected region. Returns count."""
    alignments_dbname = [
        "afdb_proteome", "afdb_swissprot", "afdb50", "cath50",
        "gmgcl_id", "mgnify_esm30", "pdb100",
    ]
    count = 0
    with open(output_fasta, "w", encoding="utf-8") as f_out:
        for db_name, filename in zip(alignments_dbname, alignments_files):
            if not os.path.exists(filename):
                continue
            parser = FoldSeekAlignmentParser(filename)
            alignments = parser.parse()
            for alignment in alignments:
                if alignment.qstart <= protect_start and alignment.qend >= protect_end:
                    f_out.write(f">{db_name} {alignment.tseqid.split(' ')[0]}\n")
                    f_out.write(f"{alignment.tseq}\n")
                    count += 1
    return count
