"""Common utility functions for VenusFactory2."""

import json
import os
import re
import tarfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote
from uuid import uuid4
import gradio as gr


def sanitize_filename(name: str) -> str:
    """Sanitize filename for safe file operations."""
    name = re.split(r'[|\s/]', name)[0]
    return re.sub(r'[^\w\-. ]', '_', name)


def get_save_path(subdir1: str, subdir2: str | None = None) -> Path:
    """Get save path with date-based directory structure. Root from env TEMP_OUTPUTS_DIR."""
    base = os.getenv("TEMP_OUTPUTS_DIR", "temp_outputs")
    now = datetime.now()
    date_path = Path(base) / f"{now.year}/{now.month:02d}/{now.day:02d}"

    if subdir2:
        path = date_path / subdir1 / subdir2
    else:
        path = date_path / subdir1

    path.mkdir(parents=True, exist_ok=True)
    return path


_WEB_V2_AREAS = {"uploads", "work", "results", "sessions", "manifests", "cache", "agent"}
_WEB_V2_ALLOWED_KINDS = {"upload", "work", "result", "manifest", "artifact"}
_ABS_PATH_PATTERN = re.compile(r"/(?:home|Users|tmp|var|opt|mnt|data)(?:/[^/\s]+)+")


def get_temp_outputs_base_dir() -> Path:
    """Get TEMP_OUTPUTS_DIR as an absolute path."""
    return Path(os.getenv("TEMP_OUTPUTS_DIR", "temp_outputs")).resolve()


def get_project_root() -> Path:
    """Resolve repository root based on src/web/utils location."""
    return Path(__file__).resolve().parent.parent.parent.parent


def to_project_relative_path(path: str | Path) -> str:
    """Convert a path to a project-relative path whenever possible."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        return target.as_posix()

    resolved = target.resolve()
    project_root = get_project_root().resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError:
        return resolved.name


def redact_path_text(text: str | None) -> str:
    """Redact absolute path segments from user-facing text."""
    raw = str(text or "")
    if not raw:
        return raw

    project_root = get_project_root().resolve().as_posix()
    normalized = raw.replace("\\", "/")
    if project_root in normalized:
        normalized = normalized.replace(project_root, ".")

    return _ABS_PATH_PATTERN.sub(
        lambda m: to_project_relative_path(m.group(0)) if m.group(0).startswith("/") else m.group(0),
        normalized,
    )


def get_dated_temp_root(now: Optional[datetime] = None, use_utc: bool = False) -> Path:
    """Get date-partitioned root under TEMP_OUTPUTS_DIR."""
    if now is not None:
        curr = now
    elif use_utc:
        curr = datetime.now(timezone.utc)
    else:
        curr = datetime.now()
    return get_temp_outputs_base_dir() / f"{curr.year}/{curr.month:02d}/{curr.day:02d}"


def build_run_id_utc(now: Optional[datetime] = None) -> str:
    """Generate run id with UTC timestamp and random suffix."""
    curr = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
    return f"{curr.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"


def generate_run_id(now: Optional[datetime] = None) -> str:
    """Backward-compatible alias for UTC run id."""
    return build_run_id_utc(now=now)


def get_web_v2_root_dir() -> Path:
    """Get Web v2 root directory under TEMP_OUTPUTS_DIR."""
    root = get_temp_outputs_base_dir() / "web_v2"
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_tool_slug(tool: str) -> str:
    """Normalize tool identifier to stable slug."""
    slug = sanitize_filename((tool or "shared").strip().lower()).replace(" ", "_")
    return slug or "shared"


def get_web_v2_area_dir(
    area: str,
    tool: str | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    now: Optional[datetime] = None,
) -> Path:
    """Build Web v2 directory under TEMP_OUTPUTS_DIR/web_v2 with UTC date partition."""
    normalized_area = (area or "").strip().lower()
    if normalized_area == "agent":
        normalized_area = "sessions"
    if normalized_area not in _WEB_V2_AREAS:
        raise ValueError(f"Invalid web_v2 area: {area}")
    if normalized_area == "agent":
        normalized_area = "sessions"

    curr = now.astimezone(timezone.utc) if now else datetime.now(timezone.utc)
    date_parts = (f"{curr.year}", f"{curr.month:02d}", f"{curr.day:02d}")
    path = get_web_v2_root_dir() / normalized_area

    if normalized_area == "sessions":
        sid = sanitize_filename((session_id or "anonymous").strip())
        path = path / sid / date_parts[0] / date_parts[1] / date_parts[2]
    elif normalized_area == "cache":
        domain = get_tool_slug(tool or "shared")
        path = path / domain
    else:
        if tool:
            path = path / get_tool_slug(tool) / date_parts[0] / date_parts[1] / date_parts[2]
            if run_id:
                path = path / sanitize_filename(run_id)

    path.mkdir(parents=True, exist_ok=True)
    return path


def make_web_v2_filename(run_id: str, kind: str, original_name: str) -> str:
    """Build normalized artifact filename from run id and kind."""
    normalized_kind = (kind or "").strip().lower()
    if normalized_kind not in _WEB_V2_ALLOWED_KINDS:
        normalized_kind = "artifact"
    safe_name = sanitize_filename(Path(original_name or "artifact.bin").name)
    return f"{run_id}__{normalized_kind}__{safe_name}"


def make_web_v2_upload_name(seq: int, original_name: str) -> str:
    """Build upload filename as u_<seq>__<safeOriginalName>."""
    safe_name = sanitize_filename(Path(original_name or "upload.bin").name)
    return f"u_{max(seq, 1):03d}__{safe_name}"


def make_web_v2_result_name(kind: str, seq: int, extension: str = "") -> str:
    """Build result filename as r_<kind>__<seq>.<ext>."""
    normalized_kind = sanitize_filename((kind or "artifact").strip().lower())
    ext = extension or ""
    if ext and not ext.startswith("."):
        ext = f".{ext}"
    return f"r_{normalized_kind}__{max(seq, 1):03d}{ext}"


def ensure_within_roots(path: Path, allowed_roots: Iterable[Path]) -> bool:
    """Check whether path is under any allowed root after resolve."""
    try:
        target = path.resolve()
    except Exception:
        return False
    for root in allowed_roots:
        try:
            resolved_root = Path(root).resolve()
            target.relative_to(resolved_root)
            return True
        except Exception:
            continue
    return False


def to_web_v2_public_path(path: str | Path) -> str:
    """Convert absolute path to web_v2-rooted relative public path."""
    target = Path(path).resolve()
    root = get_web_v2_root_dir().resolve()
    return target.relative_to(root).as_posix()


def resolve_web_v2_client_path(
    value: str | Path,
    *,
    allowed_areas: tuple[str, ...] = ("uploads", "results", "work", "sessions", "manifests", "cache"),
) -> Path:
    """Resolve a client-provided path token into a safe absolute path under web_v2."""
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("Empty path value")
    root = get_web_v2_root_dir().resolve()
    allowed_roots = [(root / area).resolve() for area in allowed_areas]

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        resolved = (root / raw.lstrip("/")).resolve()
    if not ensure_within_roots(resolved, allowed_roots):
        raise ValueError("Path is outside allowed web_v2 areas")
    return resolved


def build_web_v2_download_url(path: str | Path) -> str:
    """Build safe /api/download URL from a web_v2 path."""
    rel = to_web_v2_public_path(path) if Path(path).is_absolute() else str(path).lstrip("/")
    return f"/api/download/{quote(rel, safe='/')}"


def stage_file_for_web_v2_result(
    source_file: str | Path,
    *,
    tool: str,
    run_id: str,
    kind: str = "result",
) -> Path:
    """Copy source artifact into standardized web_v2 results tree."""
    src = Path(source_file)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Source artifact not found: {src}")
    out_dir = get_web_v2_area_dir("results", tool=tool, run_id=run_id)
    dst = out_dir / make_web_v2_filename(run_id, kind, src.name)
    dst.write_bytes(src.read_bytes())
    return dst


def create_run_manifest(
    *,
    run_id: str,
    tool: str,
    status: str = "created",
    session_id: str = "",
    inputs: Optional[list[Dict[str, Any]]] = None,
    outputs: Optional[list[Dict[str, Any]]] = None,
    expires_at_utc: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Create run manifest json under web_v2/manifests/YYYY/MM/DD."""
    manifest_dir = get_web_v2_area_dir("manifests", tool=tool)
    manifest_path = manifest_dir / f"{sanitize_filename(run_id)}.json"
    payload: Dict[str, Any] = {
        "run_id": run_id,
        "tool": get_tool_slug(tool),
        "session_id": session_id or "",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "expires_at_utc": expires_at_utc or "",
    }
    if extra:
        payload["extra"] = extra
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def cleanup_web_v2_temp_outputs() -> Dict[str, int]:
    """Optional TTL cleanup for web_v2 folders. Disabled by default."""
    enabled = os.getenv("TEMP_OUTPUTS_ENABLE_TTL_CLEANUP", "0").strip() == "1"
    if not enabled:
        return {"deleted_files": 0, "deleted_dirs": 0}

    def _safe_ttl(name: str, default: int) -> int:
        try:
            return max(int(os.getenv(name, str(default))), 0)
        except ValueError:
            return max(default, 0)

    ttl_map = {
        "uploads": _safe_ttl("TEMP_OUTPUTS_UPLOADS_TTL_DAYS", 7),
        "work": _safe_ttl("TEMP_OUTPUTS_WORK_TTL_DAYS", 2),
        "results": _safe_ttl("TEMP_OUTPUTS_RESULTS_TTL_DAYS", 14),
        "sessions": _safe_ttl("TEMP_OUTPUTS_SESSIONS_TTL_DAYS", _safe_ttl("TEMP_OUTPUTS_AGENT_TTL_DAYS", 3)),
        "manifests": _safe_ttl("TEMP_OUTPUTS_MANIFESTS_TTL_DAYS", 30),
        "cache": _safe_ttl("TEMP_OUTPUTS_CACHE_TTL_DAYS", 7),
    }
    now_ts = datetime.now(timezone.utc).timestamp()
    deleted_files = 0
    deleted_dirs = 0

    web_v2_root = get_web_v2_root_dir()
    for area, ttl_days in ttl_map.items():
        area_dir = web_v2_root / area
        if not area_dir.exists():
            continue
        cutoff = now_ts - (ttl_days * 86400)
        for item in area_dir.rglob("*"):
            try:
                stat = item.stat()
            except OSError:
                continue
            if stat.st_mtime > cutoff:
                continue
            if item.is_file():
                try:
                    item.unlink(missing_ok=True)
                    deleted_files += 1
                except OSError:
                    continue
        for sub in sorted(area_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if sub.is_dir():
                try:
                    sub.rmdir()
                    deleted_dirs += 1
                except OSError:
                    pass
    return {"deleted_files": deleted_files, "deleted_dirs": deleted_dirs}


def toggle_ai_section(is_checked: bool):
    """Toggle visibility of AI configuration section."""
    return gr.update(visible=is_checked)


def create_tar_archive(files_to_archive: Dict[str, str], output_filename: str) -> str:
    """
    Create tar.gz archive with specified files.
    
    Args:
        files_to_archive: Dict mapping source paths to archive names (internal paths)
        output_filename: The output filename (e.g., 'archive.tar.gz')
    """
    with tarfile.open(output_filename, "w:gz") as tar:
        for src, arcname in files_to_archive.items():
            if os.path.exists(src):
                tar.add(src, arcname=arcname)
            else:
                print(f"Warning: File not found: {src}")
                
    return output_filename


def format_physical_chemical_properties(data: dict) -> str:
    """Format physical and chemical properties results."""
    result = ""
    result += f"Sequence length: {data['sequence_length']} aa\n"
    result += f"Molecular weight: {data['molecular_weight'] / 1000:.2f} kDa\n"
    result += f"Theoretical pI: {data['theoretical_pI']}\n"
    result += f"Aromaticity: {data['aromaticity']}\n"
    result += f"Instability index: {data['instability_index']}\n"
    
    if data['instability_index'] > 40:
        result += "  ⚠️ Predicted as unstable protein\n"
    else:
        result += "  ✅ Predicted as stable protein\n"
    
    result += f"GRAVY: {data['gravy']}\n"
    
    ssf = data['secondary_structure_fraction']
    result += f"Secondary structure prediction: Helix={ssf['helix']}, Turn={ssf['turn']}, Sheet={ssf['sheet']}\n"
    
    return result


def format_rsa_results(data: dict) -> str:
    """Format RSA calculation results."""
    result = ""
    result += f"Exposed residues: {data['exposed_residues']}\n"
    result += f"Buried residues: {data['buried_residues']}\n"
    result += f"Total residues: {data['total_residues']}\n"
    
    def get_residue_number_rsa(item):
        """Get residue number from item for sorting."""
        return int(item[0])
    
    try:
        sorted_residues = sorted(data['residue_rsa'].items(), key=get_residue_number_rsa)
    except ValueError:
        sorted_residues = sorted(data['residue_rsa'].items())
    
    for res_id, res_data in sorted_residues:
        aa = res_data['aa']
        rsa = res_data['rsa']
        location = "Exposed (surface)" if rsa >= 0.25 else "Buried (core)"
        result += f"  Residue {res_id} ({aa}): RSA = {rsa:.3f} ({location})\n"
    return result


def format_sasa_results(data: dict) -> str:
    """Format SASA calculation results."""
    result = ""
    result += f"{'Chain':<6} {'Residue':<12} {'SASA (Ų)':<15}\n"
    
    for chain_id, chain_data in sorted(data['chains'].items()):
        result += f"--- Chain {chain_id} (Total SASA: {chain_data['total_sasa']:.2f} Ų) ---\n"
        
        def get_residue_number_sasa(item):
            """Get residue number from item for sorting."""
            return int(item[0])
        
        try:
            sorted_residues = sorted(chain_data['residue_sasa'].items(), key=get_residue_number_sasa)
        except ValueError:
            sorted_residues = sorted(chain_data['residue_sasa'].items())
        
        for res_num, res_data in sorted_residues:
            res_id_str = f"{res_data['resname']}{res_num}"
            result += f"{chain_id:<6} {res_id_str:<12} {res_data['sasa']:<15.2f}\n"

    return result


def format_secondary_structure_results(data: dict) -> str:
    """Format secondary structure calculation results."""
    result = f"Successfully calculated secondary structure for chain '{data['chain_id']}'\n"
    result += f"Sequence length: {len(data['aa_sequence'])}\n"
    result += f"Helix (H): {data['ss_counts']['helix']} ({data['ss_counts']['helix']/len(data['aa_sequence'])*100:.1f}%)\n"
    result += f"Sheet (E): {data['ss_counts']['sheet']} ({data['ss_counts']['sheet']/len(data['aa_sequence'])*100:.1f}%)\n"
    result += f"Coil (C): {data['ss_counts']['coil']} ({data['ss_counts']['coil']/len(data['aa_sequence'])*100:.1f}%)\n"
    
    def get_residue_number_ss(item):
        """Get residue number from item for sorting."""
        return int(item[0])
    
    try:
        sorted_residues = sorted(data['residue_ss'].items(), key=get_residue_number_ss)
    except ValueError:
        sorted_residues = sorted(data['residue_ss'].items())
    
    for res_id, res_data in sorted_residues:
        result += f"  Residue {res_id} ({res_data['aa_seq']}): ss8: {res_data['ss8_seq']} ({res_data['ss8_name']}), ss3: {res_data['ss3_seq']}\n"
    
    result += f"aa_seq: {data['aa_sequence']}\n"
    result += f"ss8_seq: {data['ss8_sequence']}\n"
    result += f"ss3_seq: {data['ss3_sequence']}\n"
    
    return result

