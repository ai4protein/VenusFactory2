"""Calculation workflows consumed by the API; no Gradio dependency."""
import os
import sys
import subprocess
import time
import json
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Generator, Optional, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from mcp_tools.common.constants import VENUSMINE_CONFIG
from mcp_tools.common.common_utils import create_tar_archive, get_save_path
from mcp_tools.common.file_handlers import extract_sequence_from_pdb
from mcp_tools.common.prediction_runners import PROJECT_ROOT
from mcp_tools.common.report_fragments import create_progress_html, create_status_html
from mcp_tools.common.results import result_patch
from mcp_tools.common.common_utils import get_project_root
PROJECT_ROOT = get_project_root()
import torch
from transformers import T5Tokenizer, T5EncoderModel
from mcp_tools.tools.discovery.venusmine import *
logger = logging.getLogger(__name__)
UNIREF90_FASTA_URL = os.getenv("VENUSMINE_UNIREF90_FASTA_URL", "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz").strip()
UNIREF90_DEFAULT_PATH = PROJECT_ROOT / "data" / "VenusMine" / "uniref90.fasta.gz"

def _download_file_atomic(url: str, destination: Path) -> None:
    """Download a large file via a temporary .part file, then atomically publish it."""
    if not url:
        raise ValueError("UniRef90 download URL is empty.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()
    logger.info("Downloading VenusMine UniRef90 MMseqs database from %s to %s", url, destination)
    with urllib.request.urlopen(url, timeout=60) as response, open(temp_path, "wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    os.replace(temp_path, destination)

def resolve_venusmine_mmseqs_database_path() -> Tuple[Path, str]:
    """Resolve MMseqs DB path with priority: env > constant.json > UniRef90 default."""
    env_candidates = [
        ("VENUSMINE_MMSEQS_DATABASE_PATH", os.getenv("VENUSMINE_MMSEQS_DATABASE_PATH")),
        ("MMSEQS_DATABASE_PATH", os.getenv("MMSEQS_DATABASE_PATH")),
    ]
    for env_key, env_val in env_candidates:
        if env_val:
            return Path(env_val).expanduser().resolve(), f"env:{env_key}"

    config_val = VENUSMINE_CONFIG.get("mmseqs_database_path")
    if isinstance(config_val, str) and config_val.strip():
        cfg_path = Path(config_val).expanduser()
        return (cfg_path if cfg_path.is_absolute() else PROJECT_ROOT / cfg_path).resolve(), "constant.json:venusmine.mmseqs_database_path"

    uniref90_path = UNIREF90_DEFAULT_PATH.resolve()
    if uniref90_path.exists():
        return uniref90_path, "default:cached-uniref90"
    _download_file_atomic(UNIREF90_FASTA_URL, uniref90_path)
    return uniref90_path, "default:auto-downloaded-uniref90"

def handle_VenusMine(
    pdb_file: Any,
    protect_start: int,
    protect_end: int,
    mmseqs_threads: int,
    mmseqs_iterations: int,
    mmseqs_max_seqs: int,
    cluster_min_seq_id: float,
    cluster_threads: int,
    top_n_threshold: int,
    evalue_threshold: float,
    progress=None
) -> Generator[Tuple[str, Any, Any, Any, Any, Any, Any, str], None, None]:
    """
    Returns:
        Generator yielding tuples of:
        (log_text, tree_image_path, labels_df, tree_download_btn, labels_download_btn, 
         zip_download_btn, tree_progress_html, status_indicator_html)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    start_time = time.time()

    if not pdb_file:
        yield (
            "❌ Error: Please upload a PDB file first.",
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html("❌ Missing PDB file", "#dc3545")
        )
        return

    # Create session-specific directory with timestamp
    from datetime import datetime
    now = datetime.now()
    session_dir = get_save_path("VenusMine", "Result")
    session_timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_content = "🚀 Initializing VenusMine pipeline...\n"
    log_content += f"{'='*30}\n"
    log_content += f"Session ID: {session_timestamp}\n"
    log_content += f"Output Directory: {session_dir}\n"
    log_content += f"Protected Region: {protect_start}-{protect_end}\n"
    log_content += "MMseqs Database: resolving UniRef90...\n"
    log_content += f"{'='*30}\n\n"

    yield (
        log_content,
        None,
        pd.DataFrame(),
        result_patch(visible=False),
        result_patch(visible=False),
        result_patch(visible=False),
        result_patch(value="", visible=False),
        create_status_html("🔵 Initializing...", "#0d6efd")
    )

    try:
        mmseqs_database_path, db_source = resolve_venusmine_mmseqs_database_path()
        log_content += f"   OK: MMseqs database ready: {mmseqs_database_path} ({db_source})\n\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html("MMseqs database ready", "#0d6efd")
        )
    except Exception as e:
        log_content += f"   ERROR: Failed to prepare UniRef90 database: {str(e)[:500]}\n"
        log_content += (
            "   Tip: Configure database path by setting VENUSMINE_MMSEQS_DATABASE_PATH\n"
            "      or adding venusmine.mmseqs_database_path in src/constant.json.\n"
            "      The default auto-download URL can be overridden with VENUSMINE_UNIREF90_FASTA_URL.\n"
        )
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="UniRef90 download failed"),
            result_patch(visible=True, value="*UniRef90 database preparation failed*"),
            result_patch(value="**Status:** UniRef90 database failed")
        )
        return

    try:
        # ==================== Step 1: Setup Directories ====================
        log_content += "📁 Step 1/9: Setting up directories...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html("🔵 Step 1/9 - Setup", "#0d6efd")
        )
        protein_name = "protein"
        foldseek_dir = get_save_path("VenusMine", "FoldSeek_Search")

        logger.info(f"Working directory: {foldseek_dir}")
        log_content += f"   ✓ Working directory created: {foldseek_dir}\n\n"

        pdb_file_path = foldseek_dir / f"{protein_name}.pdb"
        if isinstance(pdb_file, str):
            shutil.copy(pdb_file, pdb_file_path)
        else:
            with open(pdb_file_path, 'w') as f:
                content = pdb_file.read() if hasattr(pdb_file, 'read') else str(pdb_file)
                f.write(content)

        # ==================== Step 2: FoldSeek Search ====================
        log_content += "🔍 Step 2/9: Running FoldSeek structural search...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html("⏳ Running FoldSeek search...", "#0d6efd")
        )

        downloaded_files = download_foldseek_m8(str(pdb_file_path), foldseek_dir)
        logger.info(f"FoldSeek finished, results in {foldseek_dir}")
        log_content += f"   ✓ FoldSeek search completed\n"
        log_content += f"   ✓ Downloaded {len(downloaded_files)} alignment files\n\n"

        # ==================== Step 3: Parse FoldSeek Alignments ====================
        log_content += "📝 Step 3/9: Parsing FoldSeek alignments...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html("⏳ Parsing alignments...", "#0d6efd")
        )

        filename_list = [
            os.path.join(str(foldseek_dir), "alis_afdb-proteome.m8"),
            os.path.join(str(foldseek_dir), "alis_afdb-swissprot.m8"),
            os.path.join(str(foldseek_dir), "alis_afdb50.m8"),
            os.path.join(str(foldseek_dir), "alis_cath50.m8"),
            os.path.join(str(foldseek_dir), "alis_gmgcl_id.m8"),
            os.path.join(str(foldseek_dir), "alis_mgnify_esm30.m8"),
            os.path.join(str(foldseek_dir), "alis_pdb100.m8")
        ]
        alignments_collection = []
        alignments_dbname = [
            "afdb_proteome", "afdb_swissprot", "afdb50", "cath50",
            "gmgcl_id", "mgnify_esm30", "pdb100"]

        # Parse the FoldSeek result files
        for filename in filename_list:
            parser = FoldSeekAlignmentParser(filename)
            alignments = parser.parse()
            alignments_collection.append(alignments)

        total_alignments = 0
        # print results
        for i in range(len(alignments_dbname)):
            log_content += "Database: {}, found alignments: {}\n".format(alignments_dbname[i], len(alignments_collection[i]))
            total_alignments += len(alignments_collection[i])



        foldseek_dir_path = Path(foldseek_dir)
        output_file = foldseek_dir_path / f"{foldseek_dir_path.name}_foldseek.fasta"
        f_out = open(output_file, "w")
        count = 0

        for alignments_index in range(len(alignments_dbname)):
            alignments = alignments_collection[alignments_index]
            alignments_db = alignments_dbname[alignments_index]

            for alignment in alignments:
                # consider protect region
                if alignment.qstart <= protect_start and alignment.qend >= protect_end:
                    f_out.write(">" + alignments_db + " " + alignment.tseqid.split(" ")[0] + "\n")
                    f_out.write(alignment.tseq + "\n")
                    count += 1

        f_out.close()

        log_content += f"   ✓ Found {count} sequences matching protected region\n"
        log_content += f"   ✓ Sequences saved to: {output_file.name}\n\n"

        # Check if we have sequences to search
        if count == 0:
            log_content += "   ⚠️ No sequences found matching the protected region criteria.\n"
            log_content += "   💡 Try adjusting the protected region parameters (start/end positions).\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(value="⚠️ No sequences found"),
                result_patch(visible=True, value="⚠️ *No sequences matched protected region*"),
                result_patch(value="**Status:** ⚠️ No sequences to search")
            )
            return

        # ==================== Step 4: MMseqs Search ====================
        log_content += "🔎 Step 4/9: Running MMseqs2 sequence search...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html("⏳ Running MMseqs2 search...", "#0d6efd")
        )

        # Check if mmseqs is installed
        mmseqs_check = subprocess.run(['which', 'mmseqs'], capture_output=True, text=True)


        # Check if database exists
        if not mmseqs_database_path.exists():
            log_content += f"   ❌ Database file not found: {mmseqs_database_path}\n"
            log_content += (
                "   Tip: Configure database path by setting VENUSMINE_MMSEQS_DATABASE_PATH\n"
                "      or adding venusmine.mmseqs_database_path in src/constant.json.\n"
                "      By default, VenusMine auto-downloads UniRef90 to data/VenusMine/uniref90.fasta.gz.\n"
            )
            yield (
                log_content,
                None,
                pd.DataFrame(),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(value="❌ Database not found"),
                result_patch(visible=True, value="❌ *Database file missing*"),
                result_patch(value="**Status:** ❌ Database not found")
            )
            return

        mmseqs_prefix = foldseek_dir / f"{protein_name}_MEER"
        mmseqs_tsv = Path(str(mmseqs_prefix) + ".tsv")
        tmp_dir = get_save_path("VenusMine", "tmp")

        cmd_search = [
            "mmseqs", "easy-search",
            str(output_file),
            str(mmseqs_database_path),
            str(mmseqs_tsv),
            str(tmp_dir),
            "--format-output", "query,target,pident,fident,nident,alnlen,bits,tseq,evalue",
            "--threads", str(mmseqs_threads),
            "--num-iterations", str(mmseqs_iterations),
            "--max-seqs", str(mmseqs_max_seqs),
            "--search-type", "1"
        ]

        log_content += f"   • Database: {mmseqs_database_path.name} ({db_source})\n"
        log_content += f"   • Threads: {mmseqs_threads}, Iterations: {mmseqs_iterations}\n"
        log_content += f"   • Max sequences: {mmseqs_max_seqs}\n"

        try:
            result = subprocess.run(cmd_search, capture_output=True, text=True, check=True)
            log_content += f"   ✓ MMseqs search completed successfully\n"
        except subprocess.CalledProcessError as e:
            log_content += f"   ❌ MMseqs search failed with error:\n"
            log_content += f"   {e.stderr[:500]}\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(value="❌ MMseqs failed"),
                result_patch(visible=True, value="❌ *MMseqs search failed*"),
                result_patch(value="**Status:** ❌ Failed - MMseqs error")
            )
            return

        if not mmseqs_tsv.exists():
            log_content += "   ❌ MMseqs output file not created.\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(value="❌ MMseqs failed"),
                result_patch(visible=True, value="❌ *MMseqs search failed*"),
                result_patch(value="**Status:** ❌ Failed - MMseqs error")
            )
            return

        log_content += f"   ✓ MMseqs search completed\n"
        log_content += f"   ✓ Results saved to: {mmseqs_tsv.name}\n\n"

        # Convert TSV to FASTA
        log_content += "   • Converting TSV to FASTA format...\n"
        mmseqs_fasta = Path(str(mmseqs_prefix) + ".fasta")

        try:
            header, data = read_tsv(str(mmseqs_tsv))
            log_content += f"   • Loaded TSV file successfully, {len(data)} entries\n"
            write_fasta_from_tsv(header, data, str(mmseqs_fasta), evalue=None)
            log_content += f"   ✓ FASTA file created: {mmseqs_fasta.name}\n\n"
        except Exception as e:
            log_content += f"   ❌ TSV to FASTA conversion failed: {str(e)[:200]}\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(value="❌ Conversion failed"),
                result_patch(visible=True, value="❌ *Format conversion failed*"),
                result_patch(value="**Status:** ❌ Failed - Conversion error")
            )
            return

        if not mmseqs_fasta.exists():
            log_content += "   ❌ FASTA file was not created.\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(value="❌ Conversion failed"),
                result_patch(visible=True, value="❌ *Format conversion failed*"),
                result_patch(value="**Status:** ❌ Failed - Conversion error")
            )
            return

        # ==================== Step 5: MMseqs Clustering ====================
        log_content += "🧮 Step 5/9: Clustering sequences (removing redundancy)...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html("⏳ Clustering sequences...", "#0d6efd")
        )

        cluster_out = foldseek_dir / f"{protein_name}_clus2_NR_out"
        log_content += f"   • Min sequence identity: {cluster_min_seq_id}\n"
        log_content += f"   • Threads: {cluster_threads}\n"

        try:
            result = subprocess.run(
                ["mmseqs", "easy-cluster", str(mmseqs_fasta), str(cluster_out), str(tmp_dir),
                 "--min-seq-id", str(cluster_min_seq_id), "--threads", str(cluster_threads)],
                capture_output=True, text=True, check=True
            )
            log_content += f"   ✓ Clustering completed\n"
        except subprocess.CalledProcessError as e:
            log_content += f"   ❌ Clustering failed: {e.stderr[:200]}\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(value="❌ Clustering failed"),
                result_patch(visible=True, value="❌ *Clustering failed*"),
                result_patch(value="**Status:** ❌ Failed - Clustering error")
            )
            return

        cluster_rep_fasta = foldseek_dir / f"{protein_name}_clus2_NR_out_rep_seq.fasta"
        if not cluster_rep_fasta.exists():
            log_content += "   ❌ Cluster representative file not created.\n"
            yield (
                log_content,
                None,
                pd.DataFrame(),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(visible=False),
                result_patch(value="❌ Clustering failed"),
                result_patch(visible=True, value="❌ *Clustering failed*"),
                result_patch(value="**Status:** ❌ Failed - Clustering error")
            )
            return

        log_content += f"   ✓ Representative sequences: {cluster_rep_fasta.name}\n\n"

        # ==================== Step 6: ProstT5 Embedding ====================
        log_content += "🧬 Step 6/9: Computing ProstT5 embeddings for discovered sequences...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html("⏳ Computing embeddings...", "#0d6efd")
        )

        log_content += f"   • Loading ProstT5 model on cuda...\n"
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to("cuda")
        model = model.half()

        data = parse_fasta(str(cluster_rep_fasta))
        batch_size = 16
        log_content += f"   • Processing {len(data)} sequences (batch_size={batch_size})...\n"

        seq_labels, seq_strs, rep = calculate_representation(model, tokenizer, data, logger, start_time, batch_size=batch_size)

        result_pkl = foldseek_dir / "result.pkl"
        save_representation(seq_labels, seq_strs, rep, str(result_pkl))

        log_content += f"   ✓ Embeddings computed for {len(seq_labels)} sequences\n"
        log_content += f"   ✓ Saved to: {result_pkl.name}\n\n"

        del model, tokenizer
        torch.cuda.empty_cache()

        # ==================== Step 7: Reference (PDB) Embedding ====================
        log_content += "🎯 Step 7/9: Computing embeddings for reference protein...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html("⏳ Processing reference...", "#0d6efd")
        )

        ref_sequences = extract_sequence_from_pdb(str(pdb_file_path))
        ref_pkl = foldseek_dir / "refseq.pkl"

        log_content += f"   • Loading ProstT5 model...\n"
        tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to("cuda")
        model = model.half()

        # Reference sequences are usually few, use batch_size=1
        ref_labels, ref_strs, ref_rep = calculate_representation(model, tokenizer, ref_sequences, logger, start_time, batch_size=1)
        save_representation(ref_labels, ref_strs, ref_rep, str(ref_pkl))

        log_content += f"   ✓ Reference embeddings computed\n"
        log_content += f"   ✓ Saved to: {ref_pkl.name}\n\n"

        del model, tokenizer

        torch.cuda.empty_cache()

        # ==================== Step 8: EC Dataset Embedding ====================
        log_content += "📚 Step 8/9: Processing EC enzyme classification dataset...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html("⏳ Processing EC dataset...", "#0d6efd")
        )

        ec_fasta = PROJECT_ROOT / "data" / "VenusMine" / "ec_dataset.fasta"
        ec_pkl = PROJECT_ROOT / "data" / "VenusMine" / "ec.pkl"
        ec_csv = PROJECT_ROOT / "data" / "VenusMine" / "ec_dataset.csv"

        if not ec_pkl.exists():
            log_content += f"   • EC embeddings not found, computing...\n"
            tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
            model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to("cuda")
            model = model.half()

            data = parse_fasta(str(ec_fasta))
            batch_size = 16 if len(data) > 100 else 8
            log_content += f"   • Processing {len(data)} EC sequences (batch_size={batch_size})...\n"
            ec_labels, ec_strs, ec_rep = calculate_representation(model, tokenizer, data, logger, start_time, batch_size=batch_size)
            save_representation(ec_labels, ec_strs, ec_rep, str(ec_pkl))

            log_content += f"   ✓ EC embeddings computed and cached\n\n"

            del model, tokenizer
            torch.cuda.empty_cache()
        else:
            log_content += f"   ✓ Using cached EC embeddings\n\n"

        # ==================== Step 9: Build Tree ====================
        log_content += "🌳 Step 9/9: Building phylogenetic tree and visualization...\n"
        yield (
            log_content,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value=create_progress_html("Building phylogenetic tree..."), visible=True),
            create_status_html("🔵 Step 9/9 - Tree construction", "#0d6efd")
        )

        log_content += f"   • Top N threshold: {top_n_threshold}\n"
        log_content += f"   • E-value threshold: {evalue_threshold}\n"

        plot_path, label_path = build_and_visualize_tree(
            refseq_pkl=str(ref_pkl),
            discovered_pkl=str(result_pkl),
            ec_pkl=str(ec_pkl),
            ec_csv=str(ec_csv),
            top_n_threshold=top_n_threshold,
            output_dir=session_dir
        )

        log_content += f"   ✓ Tree visualization created\n"
        log_content += f"   ✓ Plot saved to: {Path(plot_path).name}\n"
        log_content += f"   ✓ Labels saved to: {Path(label_path).name}\n\n"

        # ==================== Final: Package Results ====================
        log_content += "📦 Packaging results...\n"

        # 读取标签数据
        labels_df = pd.DataFrame()
        if label_path and Path(label_path).exists():
            try:
                labels_df = pd.read_csv(label_path, sep='\t')  # TSV file
                log_content += f"   ✓ Loaded {len(labels_df)} sequence labels\n"
            except Exception as e:
                log_content += f"   ⚠ Could not read labels: {e}\n"

        # 创建 tar.gz 文件
        tar_path = None
        try:
            results_dir = get_save_path("VenusMine", "Result")
            files_to_tar = {}
            if plot_path and Path(plot_path).exists():
                files_to_tar[str(plot_path)] = Path(plot_path).name
            if label_path and Path(label_path).exists():
                files_to_tar[str(label_path)] = Path(label_path).name

            if files_to_tar:
                tar_path = results_dir / f"venusmine_results_{session_timestamp}.tar.gz"
                create_tar_archive(files_to_tar, str(tar_path))
                log_content += f"   ✓ Results packaged to: {tar_path.name}\n"
            else:
                log_content += "   ⚠ No files available to package\n"
        except Exception as e:
            log_content += f"   ⚠ Could not create tar.gz: {e}\n"

        # 最终日志
        log_content += f"\n{'='*60}\n"
        log_content += f"✅ VenusMine Pipeline Completed Successfully!\n"
        log_content += f"{'='*60}\n\n"
        log_content += f"📊 Summary Statistics:\n"
        log_content += f"   • Total sequences discovered: {len(labels_df)}\n"
        log_content += f"   • Processing time: {time.time() - start_time:.2f} seconds\n"
        log_content += f"   • Results directory: {session_dir}\n"
        log_content += f"\n{'='*60}\n"

        # 最终输出
        yield (
            log_content,
            str(plot_path) if plot_path else None,  # 返回图片路径
            labels_df,
            result_patch(visible=True, value=str(plot_path)),
            result_patch(visible=True, value=str(label_path)),
            result_patch(visible=True, value=str(tar_path) if tar_path else None),
            result_patch(value="", visible=False),  # 隐藏进度指示器
            create_status_html("✅ Completed successfully!", "#198754")
        )

    except Exception as e:
        # 错误处理
        error_log = f"\n{'='*60}\n"
        error_log += f"❌ Pipeline Error\n"
        error_log += f"{'='*60}\n\n"
        error_log += f"Error Type: {type(e).__name__}\n"
        error_log += f"Error Message: {str(e)}\n\n"

        import traceback
        error_log += f"Full Traceback:\n{traceback.format_exc()}\n"
        error_log += f"{'='*60}\n"

        final_log = log_content + error_log

        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)

        yield (
            final_log,
            None,
            pd.DataFrame(),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(visible=False),
            result_patch(value="", visible=False),
            create_status_html(f"❌ Error: {type(e).__name__}", "#dc3545")
        )
