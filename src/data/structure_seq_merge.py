"""
Generate missing structure sequences from pdb_dir and merge with HF dataset.
When HF dataset lacks stru_token_* (ProSST), foldseek_seq, ss8_seq, or esm3_structure_seq,
this module generates them from PDB files and merges with the dataset.

PDB naming can vary (IPR000126_A4Y3F4.pdb, A4Y3F4.pdb, A4Y3F4.ef.pdb). Matching uses
containment: a PDB is included if its filename stem contains (or is contained by) the
dataset's merge key. Merge uses the same containment logic.

VenusX: set structure_merge_key_format="{interpro_id}-{uid}" when uid and interpro_id
are separate columns. Only PDB files matching dataset samples are processed.
"""
import os
import re
import tempfile
import shutil
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from datasets import Dataset, DatasetDict

from .get_foldseek_structure_seq import get_foldseek_structure_seq
from .get_secondary_structure_seq import get_secondary_structure_seq
from .get_prosst_str_token import convert_predict_results
from .prosst.structure.get_sst_seq import SSTPredictor


def _pdb_stem(name: str) -> str:
    """Get PDB identifier from filename (without .pdb / .ef.pdb)."""
    for ext in (".ef.pdb", ".pdb"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _norm_for_match(s: str) -> str:
    """Normalize for containment matching: treat _ and - as equivalent."""
    return str(s).replace("_", "-")


def _key_matches_stem(key: str, stem: str) -> bool:
    """
    PDB stem matches dataset key if one contains the other (after normalizing _ to -).
    Handles varied naming: IPR000126_A4Y3F4.pdb, A4Y3F4.pdb, A4Y3F4.ef.pdb.
    """
    k, s = _norm_for_match(key), _norm_for_match(stem)
    return k in s or s in k or k == s


def _find_best_stem_for_key(key: str, stems: List[str]) -> Optional[str]:
    """Pick best matching PDB stem for a dataset key. Prefer exact, then key in stem."""
    k_norm = _norm_for_match(key)
    candidates = [(s, _norm_for_match(s)) for s in stems if _key_matches_stem(key, s)]
    if not candidates:
        return None
    for s, s_norm in candidates:
        if k_norm == s_norm:
            return s
    for s, s_norm in candidates:
        if k_norm in s_norm:
            return s
    return candidates[0][0]


def _list_pdb_files(pdb_dir: str) -> List[Tuple[str, str]]:
    """List (rel_path, stem) for all PDB files in pdb_dir (recursive)."""
    out = []
    for root, _, files in os.walk(pdb_dir):
        for f in files:
            if f.endswith(".pdb") or f.endswith(".ef.pdb"):
                rel = os.path.relpath(os.path.join(root, f), pdb_dir)
                out.append((rel, _pdb_stem(f)))
    return out


def _list_pdb_paths(pdb_dir: str) -> List[str]:
    """List full paths of all PDB files in pdb_dir (recursive)."""
    out = []
    for root, _, files in os.walk(pdb_dir):
        for f in files:
            if f.endswith(".pdb") or f.endswith(".ef.pdb"):
                out.append(os.path.join(root, f))
    return sorted(out)


def _collect_required_pdb_names(
    dataset_dict: DatasetDict,
    merge_key: Optional[str],
    merge_key_format: Optional[str],
) -> Set[str]:
    """Collect set of PDB identifiers (merge keys) from dataset splits."""
    names = set()
    for split in ["train", "validation", "test"]:
        if split not in dataset_dict:
            continue
        df = pd.DataFrame(dataset_dict[split])
        if merge_key_format:
            series = df.apply(
                lambda row: merge_key_format.format(
                    **{p: row[p] for p in re.findall(r"\{(\w+)\}", merge_key_format)}
                ),
                axis=1,
            )
        else:
            if merge_key not in df.columns:
                continue
            series = df[merge_key]
        names.update(series.astype(str).tolist())
    return names


def _make_filtered_pdb_dir(
    pdb_dir: str, required_names: Set[str], logger=None
) -> Tuple[str, bool]:
    """
    Filter PDBs by containment: include if stem contains (or is contained by) any dataset key.
    Handles varied naming: IPR000126_A4Y3F4.pdb, A4Y3F4.pdb, A4Y3F4.ef.pdb.
    Returns (path_to_use, is_temp). Uses recursive walk for subdirs.
    """
    all_pairs = _list_pdb_files(pdb_dir)
    all_stems = [s for _, s in all_pairs]
    to_include = {
        stem for stem in all_stems
        if any(_key_matches_stem(k, stem) for k in required_names)
    }
    matched_keys = {k for k in required_names if any(_key_matches_stem(k, s) for s in all_stems)}
    missing = required_names - matched_keys
    if missing and logger:
        examples = list(missing)[:3]
        logger.warning(
            f"PDB dir: no file matches {len(missing)} dataset identifiers (samples may be dropped). "
            f"Examples: {examples}"
        )
    if not to_include:
        raise FileNotFoundError(
            f"No PDB files in {pdb_dir} match the {len(required_names)} dataset identifiers"
        )
    if len(to_include) >= len(all_stems) * 0.95:
        if logger:
            logger.info(f"Using full pdb_dir ({len(all_stems)} files)")
        return pdb_dir, False
    cache_root = os.path.join(os.getcwd(), ".cache")
    os.makedirs(cache_root, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="venusfactory_pdb_", dir=cache_root)
    abs_pdb_dir = os.path.abspath(pdb_dir)
    for rel_path, stem in all_pairs:
        if stem in to_include:
            src = os.path.join(abs_pdb_dir, rel_path)
            dst = os.path.join(tmp, rel_path)
            d = os.path.dirname(dst)
            if d:
                os.makedirs(d, exist_ok=True)
            os.symlink(src, dst)
    if logger:
        logger.info(f"Processing {len(to_include)} / {len(all_stems)} PDB files (filtered by dataset)")
    return tmp, True


def get_required_structure_columns(args) -> List[Tuple[str, str]]:
    """
    Return list of (column_name, generator_type) that the model needs.
    generator_type: 'prosst' | 'foldseek' | 'ss8' | 'esm3'
    """
    required = []
    if "ProSST" in args.plm_model:
        vocab = args.plm_model.split("-")[1]
        required.append((f"stru_token_{vocab}", "prosst"))
    for seq_type in (args.structure_seq or []):
        if seq_type == "foldseek_seq":
            required.append((seq_type, "foldseek"))
        elif seq_type == "ss8_seq":
            required.append((seq_type, "ss8"))
        elif seq_type == "esm3_structure_seq":
            required.append((seq_type, "esm3"))
    return required


def _generate_prosst(pdb_dir: str, structure_vocab_size: int, **kwargs) -> pd.DataFrame:
    import torch
    processor = SSTPredictor(structure_vocab_size=structure_vocab_size, **kwargs)
    pdb_files = _list_pdb_paths(pdb_dir)
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {pdb_dir}")
    results = processor.predict_from_pdb(pdb_files)
    rows = convert_predict_results(results, structure_vocab_size)
    df = pd.DataFrame(rows)
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def _generate_foldseek(pdb_dir: str) -> pd.DataFrame:
    results = get_foldseek_structure_seq(pdb_dir)
    return pd.DataFrame(results)


def _generate_ss8(pdb_dir: str, num_workers: int = 4) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm

    pdb_files = _list_pdb_paths(pdb_dir)
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {pdb_dir}")
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(get_secondary_structure_seq, f) for f in pdb_files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="ss8"):
            r, err = future.result()
            if err is None:
                results.append(r)
    return pd.DataFrame(results)[["name", "ss8_seq"]]


def _generate_esm3(pdb_dir: str, device: str = "cuda:0") -> pd.DataFrame:
    import torch
    from .get_esm3_structure_seq import get_esm3_structure_seq, ESM3_structure_encoder_v0

    encoder = ESM3_structure_encoder_v0(device)
    pdb_files = _list_pdb_paths(pdb_dir)
    if not pdb_files:
        raise FileNotFoundError(f"No PDB files found in {pdb_dir}")
    results = []
    for f in pdb_files:
        r = get_esm3_structure_seq(f, encoder, device)
        r["name"] = os.path.basename(f).split(".")[0]
        results.append(r)
    df = pd.DataFrame(results)[["name", "esm3_structure_seq"]]
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def generate_structure_data(
    pdb_dir: str,
    columns_to_generate: List[Tuple[str, str]],
    args,
    logger=None,
    cleanup_temp_dir: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Generate structure data for missing columns. Returns {column_name: DataFrame}."""
    try:
        out = {}
        for col, gen_type in columns_to_generate:
            if logger:
                logger.info(f"Generating {col} from {pdb_dir} ({gen_type})...")
            if gen_type == "prosst":
                vocab = int(col.replace("stru_token_", ""))
                kwargs = {
                    "num_processes": getattr(args, "prosst_num_processes", 6),
                    "num_threads": getattr(args, "prosst_num_threads", 8),
                    # Default 2000 to avoid GPU OOM (ProSST default 10000 batches too many nodes)
                    # 2000 is good for 10G GPU VRAM
                    "max_batch_nodes": getattr(args, "prosst_max_batch_nodes", None) or 2000,
                }
                if getattr(args, "prosst_device", None) is not None:
                    kwargs["device"] = args.prosst_device
                df = _generate_prosst(pdb_dir, vocab, **kwargs)
                out[col] = df[["name", col]]
            elif gen_type == "foldseek":
                df = _generate_foldseek(pdb_dir)
                out[col] = df[["name", col]]
            elif gen_type == "ss8":
                df = _generate_ss8(pdb_dir, num_workers=getattr(args, "num_workers", 4))
                out[col] = df
            elif gen_type == "esm3":
                df = _generate_esm3(pdb_dir)
                out[col] = df
        return out
    finally:
        if cleanup_temp_dir and os.path.isdir(pdb_dir):
            shutil.rmtree(pdb_dir, ignore_errors=True)
        # Free GPU memory after all structure generation for subsequent training
        try:
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass


def _dataset_to_df(ds: Dataset) -> pd.DataFrame:
    return pd.DataFrame(ds)


def _df_to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df, preserve_index=False)


def _make_merge_key_column(df: pd.DataFrame, merge_key_format: str) -> pd.Series:
    """Create merge key from format string e.g. '{interpro_id}-{uid}' -> interpro_id + '-' + uid."""
    placeholders = re.findall(r"\{(\w+)\}", merge_key_format)
    for p in placeholders:
        if p not in df.columns:
            raise ValueError(f"structure_merge_key_format requires column '{p}' which is missing")
    return df.apply(lambda row: merge_key_format.format(**{p: row[p] for p in placeholders}), axis=1)


def merge_structure_into_dataset(
    dataset_dict: DatasetDict,
    structure_dfs: Dict[str, pd.DataFrame],
    merge_key: str = "name",
    merge_key_format: Optional[str] = None,
    merge_by_containment: bool = True,
    logger=None,
) -> DatasetDict:
    """
    Merge structure DataFrames into train/val/test.
    - merge_key: single column name (e.g. 'name')
    - merge_key_format: for VenusX, use '{interpro_id}-{uid}' when uid and interpro_id are separate
    - merge_by_containment: match by containment (key in PDB stem or vice versa) for varied PDB naming
    """
    merged = {}
    struct_merge_col = "name"
    for split in ["train", "validation", "test"]:
        if split not in dataset_dict:
            continue
        df = _dataset_to_df(dataset_dict[split])
        if merge_key_format:
            df = df.copy()
            df["_structure_merge_key"] = _make_merge_key_column(df, merge_key_format)
            left_key_col = "_structure_merge_key"
        else:
            if merge_key not in df.columns:
                raise ValueError(f"Dataset missing merge key column '{merge_key}'")
            left_key_col = merge_key

        for col, struct_df in structure_dfs.items():
            if col in df.columns:
                continue
            before = len(df)
            if merge_by_containment:
                stems = struct_df[struct_merge_col].astype(str).tolist()
                df["_struct_match"] = df[left_key_col].apply(
                    lambda k: _find_best_stem_for_key(str(k), stems)
                )
                df = df[df["_struct_match"].notna()]
                df = df.merge(
                    struct_df[[struct_merge_col, col]],
                    left_on="_struct_match",
                    right_on=struct_merge_col,
                    how="inner",
                )
                df = df.drop(columns=["_struct_match", struct_merge_col], errors="ignore")
            else:
                df = df.merge(
                    struct_df[[struct_merge_col, col]],
                    left_on=left_key_col,
                    right_on=struct_merge_col,
                    how="inner",
                )
                if left_key_col != struct_merge_col and struct_merge_col in df.columns:
                    df = df.drop(columns=[struct_merge_col], errors="ignore")
            if logger and len(df) < before:
                logger.warning(f"  {split}: {before - len(df)} samples dropped (no structure match for {col})")
        if "_structure_merge_key" in df.columns:
            df = df.drop(columns=["_structure_merge_key"], errors="ignore")
        merged[split] = _df_to_dataset(df)
    return DatasetDict(merged)


def ensure_structure_columns(
    dataset_dict: DatasetDict,
    args,
    logger,
) -> DatasetDict:
    """
    If HF dataset lacks required structure columns, generate from pdb_dir and merge.
    Raises ValueError if columns are missing and pdb_dir is not provided.

    VenusX: set structure_merge_key_format="{interpro_id}-{uid}" in dataset config,
    since PDB filenames are interpro_id-uid but HF has uid and interpro_id as separate columns.
    """
    required = get_required_structure_columns(args)
    if not required:
        return dataset_dict

    sample = dataset_dict["train"][0]
    merge_key_format = getattr(args, "structure_merge_key_format", None)
    if merge_key_format:
        merge_key = None
    else:
        merge_key = getattr(args, "structure_merge_key", "name")
        if merge_key not in sample:
            for k in ["id", "protein_id", "identifier", "uid", "uniprot_id"]:
                if k in sample:
                    merge_key = k
                    break
            else:
                raise ValueError(f"Cannot merge structure: no '{merge_key}' or 'id' column in dataset")
    missing = [(col, gtype) for col, gtype in required if col not in sample]
    if not missing:
        return dataset_dict

    pdb_dir = getattr(args, "pdb_dir", None)
    if not pdb_dir or not os.path.isdir(pdb_dir):
        raise ValueError(
            f"Dataset is missing structure columns: {[c for c, _ in missing]}. "
            "Provide --pdb_dir with PDB files to generate them."
        )

    required_names = _collect_required_pdb_names(
        dataset_dict, merge_key, merge_key_format
    )
    work_dir, is_temp = _make_filtered_pdb_dir(pdb_dir, required_names, logger)
    if logger:
        logger.info(f"Generating {len(missing)} structure columns from {work_dir}")
    structure_dfs = generate_structure_data(
        work_dir, missing, args, logger, cleanup_temp_dir=is_temp
    )
    merged = merge_structure_into_dataset(
        dataset_dict, structure_dfs,
        merge_key=merge_key or "name",
        merge_key_format=merge_key_format,
        logger=logger,
    )
    # Clear GPU memory after graph/structure generation so training has full VRAM
    try:
        import torch
        import gc
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:
        pass
    return merged
