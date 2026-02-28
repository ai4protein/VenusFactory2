"""Archive handling: unzip, ungzip (extracted from search.database.utils)."""
import gzip
import os
import shutil
import zipfile


def unzip_archive(zip_path: str, save_folder: str) -> str:
    """Extract a zip file to save_folder. Returns save_folder path."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(save_folder)
    return save_folder


def ungzip_file(gz_path: str, out_dir: str) -> str:
    """Decompress a .gz file into out_dir. Returns path to the decompressed file."""
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.basename(gz_path)
    if base_name.endswith(".gz"):
        out_name = base_name[:-3]
    else:
        out_name = base_name + ".out"
    out_path = os.path.join(out_dir, out_name)
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    return out_path
