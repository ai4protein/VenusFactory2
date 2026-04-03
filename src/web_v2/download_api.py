import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from web.utils.common_utils import (
    build_web_v2_download_url,
    build_run_id_utc,
    create_run_manifest,
    get_web_v2_area_dir,
    make_web_v2_result_name,
    to_web_v2_public_path,
)


router = APIRouter(prefix="/api/download", tags=["download-v2"])

ONLINE_MAX_BATCH_ITEMS = 50
WEB_V2_RESULTS_ROOT = get_web_v2_area_dir("results")
DOWNLOAD_EXPORT_DIR = get_web_v2_area_dir("results", tool="download")
DOWNLOAD_EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def _runtime_mode() -> str:
    mode = os.getenv("WEBUI_V2_MODE", "local").strip().lower()
    return mode if mode in {"local", "online"} else "local"


def _local_batch_limit() -> int:
    raw = os.getenv("WEBUI_V2_LOCAL_DOWNLOAD_BATCH_LIMIT", "0").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 0
    return max(0, value)


def _effective_batch_limit() -> int | None:
    if _runtime_mode() == "online":
        return ONLINE_MAX_BATCH_ITEMS
    local_limit = _local_batch_limit()
    return local_limit if local_limit > 0 else None


class DownloadRequestBase(BaseModel):
    method: Literal["Single ID", "From File"] = "Single ID"
    id_value: str = ""
    file_content: str = ""
    save_error_file: bool = True


class SequenceDownloadRequest(DownloadRequestBase):
    merge: bool = False


class RcsbStructureDownloadRequest(DownloadRequestBase):
    file_type: Literal["pdb", "cif"] = "pdb"
    unzip: bool = True


def _normalize_script_path(script_path: str) -> str:
    return script_path if script_path.startswith("src/") else f"src/tools/database/{script_path}"


def _run_download_script(script_path: str, **kwargs: Any) -> Tuple[bool, str, str]:
    script_full = _normalize_script_path(script_path)
    cmd = [sys.executable, script_full]
    for key, value in kwargs.items():
        if value is None or value == "":
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = f"Download completed successfully:\n{result.stdout}"
        return True, output, " ".join(cmd)
    except subprocess.CalledProcessError as exc:
        output = f"Error during download script execution:\n{exc.stderr}"
        return False, output, " ".join(cmd)
    except FileNotFoundError:
        output = f"Error: Script not found at '{script_full}'. Please check the path."
        return False, output, " ".join(cmd)
    except Exception as exc:
        output = f"An unexpected error occurred: {exc}"
        return False, output, " ".join(cmd)


def _task_folder(source_name: str, run_id: str) -> Path:
    path = get_web_v2_area_dir("work", tool=f"download_{source_name.lower()}", run_id=run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tar_task_results(task_folder: Path, archive_name: str, run_id: str) -> Tuple[Path, str]:
    tar_name = make_web_v2_result_name(archive_name, 1, ".tar.gz")
    tar_original = task_folder / "archives" / tar_name
    tar_original.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_original, "w:gz") as tar:
        for root, dirs, files in os.walk(task_folder):
            if "archives" in dirs:
                dirs.remove("archives")
            for file_name in files:
                file_path = Path(root) / file_name
                arcname = file_path.relative_to(task_folder)
                tar.add(file_path, arcname=str(arcname))

    result_dir = get_web_v2_area_dir("results", tool="download", run_id=run_id)
    export_path = result_dir / tar_name
    shutil.copy2(tar_original, export_path)
    relative = export_path.relative_to(WEB_V2_RESULTS_ROOT).as_posix()
    return tar_original, relative


def _first_file_preview(task_folder: Path, max_lines: int = 10) -> str:
    if not task_folder.exists():
        return "No output folder yet."
    skip_suffixes = (".tar.gz", ".gz", "_failed.txt")
    skip_names = ("truncated_input.txt", "truncated_input.json")
    try:
        for root, dirs, files in os.walk(task_folder):
            if "archives" in dirs:
                dirs.remove("archives")
            for file_name in sorted(files):
                if file_name.endswith(skip_suffixes) or file_name in skip_names:
                    continue
                path = Path(root) / file_name
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as fp:
                        lines = fp.readlines()[:max_lines]
                        content = "".join(lines).rstrip()
                        if content:
                            return f"[{file_name}] first {max_lines} lines:\n{content}"
                except OSError as exc:
                    return f"Cannot read {file_name}: {exc}"
                except Exception as exc:
                    return f"Cannot preview {file_name}: {str(exc)[:80]}"
        return "No readable text files in output directory"
    except Exception as exc:
        return f"Error: {str(exc)[:80]}"


def _resolve_inputs(payload: DownloadRequestBase, task_folder: Path) -> Tuple[str, str, str]:
    method = payload.method
    id_value = payload.id_value.strip()
    file_content = payload.file_content

    if method == "Single ID":
        if not id_value:
            raise HTTPException(status_code=400, detail="id_value is required for Single ID mode.")
        return id_value, "", ""

    lines = [line.strip() for line in file_content.splitlines() if line.strip()]
    if not lines:
        raise HTTPException(status_code=400, detail="file_content is required for From File mode.")

    max_batch_items = _effective_batch_limit()
    original_count = len(lines)
    truncated_lines = lines[:max_batch_items] if max_batch_items else lines
    truncated = bool(max_batch_items and original_count > max_batch_items)

    file_path = task_folder / "truncated_input.txt"
    file_path.write_text("\n".join(truncated_lines) + "\n", encoding="utf-8")

    preview_lines = truncated_lines[:20]
    preview = "\n".join(preview_lines)
    if len(truncated_lines) > 20:
        preview += f"\n... and {len(truncated_lines) - 20} more entries (showing first 20)"
    if truncated:
        preview += (
            f"\n\nBATCH LIMIT: Only processing first {max_batch_items} items out of {original_count} total items."
        )
    return "", str(file_path), preview


def _build_response(
    *,
    success: bool,
    message: str,
    preview: str,
    archive_relative_path: str,
    task_folder: Path,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "success": success,
        "message": message,
        "preview": preview,
        "archive_relative_path": archive_relative_path,
        "archive_download_url": build_web_v2_download_url(archive_relative_path) if archive_relative_path else "",
        "task_folder": to_web_v2_public_path(task_folder),
        "details": details,
    }


@router.get("/meta")
async def download_meta():
    mode = _runtime_mode()
    max_batch_items = _effective_batch_limit()
    return {
        "mode": mode,
        "max_batch_items": max_batch_items,
        "online_max_batch_items": ONLINE_MAX_BATCH_ITEMS,
        "local_max_batch_items": _local_batch_limit() or None,
        "methods": ["Single ID", "From File"],
        "structure_file_types": ["pdb", "cif"],
    }


@router.post("/uniprot")
async def download_uniprot(payload: SequenceDownloadRequest):
    run_id = build_run_id_utc()
    task_folder = _task_folder("UniProt", run_id)
    id_value, file_path, input_preview = _resolve_inputs(payload, task_folder)
    timestamp = int(time.time())
    success, output, _ = _run_download_script(
        "uniprot/uniprot_sequence.py",
        uniprot_id=id_value or None,
        file=file_path or None,
        out_dir=str(task_folder),
        merge=payload.merge,
        error_file=str(task_folder / f"{timestamp}_failed.txt") if payload.save_error_file else None,
    )
    tar_original, archive_relative = _tar_task_results(task_folder, "uniprot_sequence", run_id)
    create_run_manifest(
        run_id=run_id,
        tool="download",
        status="completed" if success else "failed",
        outputs=[{"path": archive_relative}],
        extra={"source": "UniProt"},
    )
    preview = _first_file_preview(task_folder)
    return _build_response(
        success=success,
        message=output,
        preview=preview,
        archive_relative_path=archive_relative,
        task_folder=task_folder,
        details={
            "source": "UniProt",
            "method": payload.method,
            "input_preview": input_preview,
        },
    )


@router.post("/ncbi")
async def download_ncbi(payload: SequenceDownloadRequest):
    run_id = build_run_id_utc()
    task_folder = _task_folder("NCBI", run_id)
    id_value, file_path, input_preview = _resolve_inputs(payload, task_folder)
    timestamp = int(time.time())
    success, output, _ = _run_download_script(
        "ncbi/ncbi_sequence.py",
        id=id_value or None,
        file=file_path or None,
        out_dir=str(task_folder),
        merge=payload.merge,
        error_file=str(task_folder / f"{timestamp}_failed.txt") if payload.save_error_file else None,
    )
    tar_original, archive_relative = _tar_task_results(task_folder, "ncbi_sequence", run_id)
    create_run_manifest(
        run_id=run_id,
        tool="download",
        status="completed" if success else "failed",
        outputs=[{"path": archive_relative}],
        extra={"source": "NCBI"},
    )
    preview = _first_file_preview(task_folder)
    return _build_response(
        success=success,
        message=output,
        preview=preview,
        archive_relative_path=archive_relative,
        task_folder=task_folder,
        details={
            "source": "NCBI",
            "method": payload.method,
            "input_preview": input_preview,
        },
    )


@router.post("/rcsb-structure")
async def download_rcsb_structure(payload: RcsbStructureDownloadRequest):
    run_id = build_run_id_utc()
    task_folder = _task_folder("RCSB", run_id)
    id_value, file_path, input_preview = _resolve_inputs(payload, task_folder)
    timestamp = int(time.time())
    success, output, _ = _run_download_script(
        "rcsb/rcsb_structure.py",
        pdb_id=id_value or None,
        pdb_id_file=file_path or None,
        out_dir=str(task_folder),
        type=payload.file_type,
        unzip=payload.unzip,
        error_file=str(task_folder / f"{timestamp}_failed.txt") if payload.save_error_file else None,
    )
    tar_original, archive_relative = _tar_task_results(task_folder, "rcsb_structure", run_id)
    create_run_manifest(
        run_id=run_id,
        tool="download",
        status="completed" if success else "failed",
        outputs=[{"path": archive_relative}],
        extra={"source": "RCSB Structure"},
    )
    preview = _first_file_preview(task_folder)
    viz_status = "Download failed."
    if success and payload.method == "Single ID" and id_value:
        expected = task_folder / f"{id_value}.{payload.file_type}"
        if expected.exists():
            viz_status = f"Structure ready: {expected.name}"
        else:
            viz_status = f"Downloaded {id_value}, but expected structure file was not found."
    elif success:
        viz_status = "Batch download complete. Use archive for all results."
    return _build_response(
        success=success,
        message=output,
        preview=preview,
        archive_relative_path=archive_relative,
        task_folder=task_folder,
        details={
            "source": "RCSB Structure",
            "method": payload.method,
            "input_preview": input_preview,
            "visualization_status": viz_status,
        },
    )


@router.post("/alphafold-structure")
async def download_alphafold_structure(payload: DownloadRequestBase):
    run_id = build_run_id_utc()
    task_folder = _task_folder("AlphaFold", run_id)
    id_value, file_path, input_preview = _resolve_inputs(payload, task_folder)
    timestamp = int(time.time())
    success, output, _ = _run_download_script(
        "alphafold/alphafold_structure.py",
        uniprot_id=id_value or None,
        uniprot_id_file=file_path or None,
        out_dir=str(task_folder),
        error_file=str(task_folder / f"{timestamp}_failed.txt") if payload.save_error_file else None,
    )
    tar_original, archive_relative = _tar_task_results(task_folder, "alphafold_structure", run_id)
    create_run_manifest(
        run_id=run_id,
        tool="download",
        status="completed" if success else "failed",
        outputs=[{"path": archive_relative}],
        extra={"source": "AlphaFold Structure"},
    )
    preview = _first_file_preview(task_folder)
    viz_status = "Download failed."
    if success and payload.method == "Single ID" and id_value:
        expected = task_folder / f"{id_value}.pdb"
        if expected.exists():
            viz_status = f"Structure ready: {expected.name}"
        else:
            viz_status = f"Downloaded {id_value}, but expected structure file was not found."
    elif success:
        viz_status = "Batch download complete. Use archive for all results."
    return _build_response(
        success=success,
        message=output,
        preview=preview,
        archive_relative_path=archive_relative,
        task_folder=task_folder,
        details={
            "source": "AlphaFold Structure",
            "method": payload.method,
            "input_preview": input_preview,
            "visualization_status": viz_status,
        },
    )


@router.post("/rcsb-metadata")
async def download_rcsb_metadata(payload: DownloadRequestBase):
    run_id = build_run_id_utc()
    task_folder = _task_folder("RCSB", run_id)
    id_value, file_path, input_preview = _resolve_inputs(payload, task_folder)
    timestamp = int(time.time())
    success, output, _ = _run_download_script(
        "rcsb/rcsb_metadata.py",
        pdb_id=id_value or None,
        pdb_id_file=file_path or None,
        out_dir=str(task_folder),
        error_file=str(task_folder / f"{timestamp}_failed.txt") if payload.save_error_file else None,
    )
    tar_original, archive_relative = _tar_task_results(task_folder, "rcsb_metadata", run_id)
    create_run_manifest(
        run_id=run_id,
        tool="download",
        status="completed" if success else "failed",
        outputs=[{"path": archive_relative}],
        extra={"source": "RCSB Metadata"},
    )
    preview = _first_file_preview(task_folder)
    return _build_response(
        success=success,
        message=output,
        preview=preview,
        archive_relative_path=archive_relative,
        task_folder=task_folder,
        details={
            "source": "RCSB Metadata",
            "method": payload.method,
            "input_preview": input_preview,
        },
    )


@router.post("/interpro-metadata")
async def download_interpro_metadata(payload: DownloadRequestBase):
    run_id = build_run_id_utc()
    task_folder = _task_folder("InterPro", run_id)
    id_value, file_path, input_preview = _resolve_inputs(payload, task_folder)
    timestamp = int(time.time())
    success, output, _ = _run_download_script(
        "interpro/interpro_metadata.py",
        interpro_id=id_value or None,
        interpro_id_file=file_path or None,
        out_dir=str(task_folder),
        error_file=str(task_folder / f"{timestamp}_failed.txt") if payload.save_error_file else None,
    )
    tar_original, archive_relative = _tar_task_results(task_folder, "interpro_metadata", run_id)
    create_run_manifest(
        run_id=run_id,
        tool="download",
        status="completed" if success else "failed",
        outputs=[{"path": archive_relative}],
        extra={"source": "InterPro Metadata"},
    )
    preview = _first_file_preview(task_folder)
    return _build_response(
        success=success,
        message=output,
        preview=preview,
        archive_relative_path=archive_relative,
        task_folder=task_folder,
        details={
            "source": "InterPro Metadata",
            "method": payload.method,
            "input_preview": input_preview,
        },
    )
