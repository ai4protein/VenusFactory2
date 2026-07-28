"""
Skills middleware: load SKILL.md metadata and content for CB/MLS.

Discovers two roots:
  1. VenusFactory2 native: ``src/agent/skills/``
  2. scientific-agent-skills (optional submodule):
     ``third_party/scientific-agent-skills/skills/``

On name collisions, VF2 keeps the bare ``skill_id``; the scientific package is
registered as ``sas_<dirname>`` so the full upstream library stays addressable.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VF_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
_SAS_SKILLS_DIR = _REPO_ROOT / "third_party" / "scientific-agent-skills" / "skills"
# Back-compat alias used by older callers/tests.
_SKILLS_DIR = _VF_SKILLS_DIR

_SAS_PREFIX = "sas_"

# Agent Skills top-level frontmatter keys (closed set). VF2 extras go under metadata.
_ALLOWED_TOP_LEVEL = {
    "name",
    "description",
    "license",
    "compatibility",
    "allowed-tools",
    "metadata",
}


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """Parse YAML-like frontmatter between first --- and second ---.

    Handles three forms:
      key: value                       — single-line scalar
      key: >-                          — YAML folded scalar (lines joined with spaces, no final newline)
        line1
        line2
      key: >                           — YAML folded scalar (lines joined with spaces, trailing newline)
        line1
    Also collects indented `metadata:` block keys into metadata.<key>.
    """
    out: Dict[str, Any] = {}
    if not text.strip().startswith("---"):
        return out
    parts = text.split("---", 2)
    if len(parts) < 3:
        return out
    lines = parts[1].strip().splitlines()

    i = 0
    in_metadata = False
    metadata: Dict[str, str] = {}
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if in_metadata and indent > 0 and ":" in stripped:
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"").strip()
            if key:
                metadata[key] = value
            i += 1
            continue
        if in_metadata and indent == 0:
            in_metadata = False

        if ":" not in stripped:
            i += 1
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "metadata" and (not value or value in ("|", ">", ">-")):
            in_metadata = True
            i += 1
            continue
        # Folded scalar: collect indented continuation lines
        if value in (">", ">-", "|", "|-"):
            collected: List[str] = []
            base_indent = None
            j = i + 1
            while j < len(lines):
                cont = lines[j]
                if not cont.strip():
                    j += 1
                    continue
                cont_indent = len(cont) - len(cont.lstrip())
                if base_indent is None:
                    base_indent = cont_indent
                if cont_indent < base_indent or cont_indent == 0:
                    break
                collected.append(cont.strip())
                j += 1
            joined = " ".join(collected).strip().strip("'\"").strip()
            if key:
                out[key] = joined
            i = j
            continue
        # Single-line scalar
        value = value.strip("'\"").strip()
        if key and value:
            out[key] = value
        i += 1
    if metadata:
        out["metadata"] = metadata
    return out


# Whitelist aliases so nature_* manifests can load shared fragments without
# arbitrary path traversal. Legacy manifests used ``../_shared/``; the on-disk
# directory is ``_shared_nature/``.
_SHARED_PATH_ALIASES = (
    ("../_shared/", "_shared_nature/"),
    ("../_shared_nature/", "_shared_nature/"),
    ("_shared_nature/", "_shared_nature/"),
)


def _scientific_skills_dir() -> Optional[Path]:
    if _SAS_SKILLS_DIR.is_dir():
        return _SAS_SKILLS_DIR.resolve()
    return None


def _skill_roots() -> List[Tuple[str, Path]]:
    """Ordered roots: VF2 first, then scientific-agent-skills if present."""
    roots: List[Tuple[str, Path]] = []
    if _VF_SKILLS_DIR.is_dir():
        roots.append(("venusfactory", _VF_SKILLS_DIR.resolve()))
    sas = _scientific_skills_dir()
    if sas is not None:
        roots.append(("scientific", sas))
    return roots


@functools.lru_cache(maxsize=1)
def _skill_id_to_root() -> Dict[str, Path]:
    """Map public skill_id → absolute package directory."""
    mapping: Dict[str, Path] = {}
    claimed: set[str] = set()
    for source, root in _skill_roots():
        for path in sorted(root.iterdir()):
            if not path.is_dir() or path.name.startswith("_"):
                continue
            if not (path / "SKILL.md").is_file():
                continue
            dirname = path.name
            if source == "venusfactory":
                skill_id = dirname
            elif dirname in claimed:
                skill_id = f"{_SAS_PREFIX}{dirname}"
            else:
                skill_id = dirname
            # Never overwrite a VF2 (or earlier) claim.
            if skill_id in mapping:
                continue
            mapping[skill_id] = path.resolve()
            claimed.add(dirname)
            claimed.add(skill_id)
    return mapping


def resolve_skill_path(skill_id: str, relative_path: Optional[str] = None) -> Optional[Path]:
    """Resolve a path inside a skill package (or whitelisted shared roots).

    Blocks arbitrary ``..`` traversal. Allows ``../_shared/...`` and
    ``_shared_nature/...`` only when they resolve under VF2 ``skills/_shared_nature/``.
    """
    if not skill_id or skill_id.startswith("_") or "/" in skill_id or "\\" in skill_id:
        return None
    root = _skill_id_to_root().get(skill_id)
    if root is None or not root.is_dir():
        return None

    if relative_path is None or relative_path in ("", "SKILL.md"):
        path = root / "SKILL.md"
        return path if path.is_file() else None

    rel_str = relative_path.replace("\\", "/").lstrip("./")
    # Shared-nature whitelist only applies to VF2 packages.
    vf_root = _VF_SKILLS_DIR.resolve()
    if str(root).startswith(str(vf_root)):
        for prefix, mapped in _SHARED_PATH_ALIASES:
            if rel_str.startswith(prefix) or relative_path.replace("\\", "/").startswith(prefix):
                raw = relative_path.replace("\\", "/")
                for pfx, _ in _SHARED_PATH_ALIASES:
                    if raw.startswith(pfx):
                        raw = mapped + raw[len(pfx):]
                        break
                shared = (vf_root / raw).resolve()
                shared_root = (vf_root / "_shared_nature").resolve()
                if str(shared).startswith(str(shared_root)) and shared.is_file():
                    return shared
                return None

    rel = Path(rel_str)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    path = (root / rel).resolve()
    if not str(path).startswith(str(root)):
        return None
    return path if path.is_file() else None


def get_skill_root(skill_id: str) -> Optional[str]:
    """Return absolute skill package root for MLS file loading."""
    if not skill_id or skill_id.startswith("_"):
        return None
    root = _skill_id_to_root().get(skill_id)
    if root is not None and root.is_dir() and (root / "SKILL.md").is_file():
        return str(root)
    return None


def _rel_skill_md_path(skill_md: Path) -> str:
    try:
        return str(skill_md.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(skill_md)


@functools.lru_cache(maxsize=1)
def get_skills_metadata() -> List[Dict[str, Any]]:
    """
    Discover SKILL.md under VF2 + scientific-agent-skills roots.

    Each dict: skill_id, name, description, path, version, license,
    name_matches_dir, source (venusfactory|scientific).
    """
    result: List[Dict[str, Any]] = []
    id_map = _skill_id_to_root()
    # Invert root→source for tagging
    root_source: Dict[str, str] = {}
    for source, root in _skill_roots():
        root_source[str(root.resolve())] = source

    for skill_id, pkg in sorted(id_map.items(), key=lambda kv: kv[0]):
        skill_md = pkg / "SKILL.md"
        try:
            raw = skill_md.read_text(encoding="utf-8")
        except Exception:
            continue
        meta = _parse_frontmatter(raw)
        nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        name = meta.get("name", skill_id)
        source = root_source.get(str(pkg.parent.resolve()), "venusfactory")
        # pkg.parent is skills root; for VF2 pkg is under _VF_SKILLS_DIR
        if str(pkg.resolve()).startswith(str(_VF_SKILLS_DIR.resolve())):
            source = "venusfactory"
        elif _SAS_SKILLS_DIR.is_dir() and str(pkg.resolve()).startswith(str(_SAS_SKILLS_DIR.resolve())):
            source = "scientific"
        result.append({
            "skill_id": skill_id,
            "name": name,
            "description": meta.get("description", ""),
            "path": _rel_skill_md_path(skill_md),
            "version": nested.get("version") or meta.get("version", ""),
            "license": meta.get("license", ""),
            "name_matches_dir": name == skill_id or name == pkg.name,
            "source": source,
            "dirname": pkg.name,
        })
    return result


def _format_skills_metadata_string(
    *,
    max_desc: int = 800,
    sources: Optional[List[str]] = None,
    max_desc_by_source: Optional[Dict[str, int]] = None,
) -> str:
    items = get_skills_metadata()
    if sources is not None:
        allow = set(sources)
        items = [s for s in items if s.get("source") in allow]
    if not items:
        return "(No skills loaded.)"
    lines = []
    for s in items:
        sid = s.get("skill_id", "")
        desc = (s.get("description") or "")
        src = s.get("source") or "venusfactory"
        cap = max_desc
        if max_desc_by_source and src in max_desc_by_source:
            cap = max_desc_by_source[src]
        if cap > 0 and len(desc) > cap:
            desc = desc[: cap - 3] + "..."
        ver = s.get("version") or ""
        ver_bit = f" v{ver}" if ver else ""
        src_bit = " [scientific]" if src == "scientific" else ""
        lines.append(f"- **{sid}**{ver_bit}{src_bit} (skill_id: `{sid}`): {desc}")
    return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def get_skills_metadata_string() -> str:
    """Format skills metadata for Expert CB/MLS prompts.

    VF2 skills get longer descriptions; scientific reference skills stay compact
    so the prompt does not explode with ~150 packages.
    """
    vf = _format_skills_metadata_string(sources=["venusfactory"], max_desc=800)
    sas = _format_skills_metadata_string(sources=["scientific"], max_desc=120)
    if sas.startswith("(No skills"):
        return vf
    return (
        "## VenusFactory2 skills (tool-bound; prefer these for execution)\n"
        f"{vf}\n\n"
        "## Scientific-agent-skills reference (read-only knowledge; "
        "use `read_skill`; upstream scripts/uv are NOT auto-executed)\n"
        f"{sas}"
    )


@functools.lru_cache(maxsize=1)
def get_skills_catalog_for_agent() -> str:
    """Compact catalog for Science Agent system prompt (self-directed loading)."""
    vf = _format_skills_metadata_string(sources=["venusfactory"], max_desc=220)
    sas = _format_skills_metadata_string(sources=["scientific"], max_desc=80)
    if sas.startswith("(No skills"):
        return vf
    return (
        "## VenusFactory2 skills (prefer for tool execution)\n"
        f"{vf}\n\n"
        "## Scientific-agent-skills reference library "
        "(call `read_skill` / `list_skills` for details; "
        "name collisions use `sas_<id>`)\n"
        f"{sas}"
    )


def get_skill_content(skill_id: str, relative_path: Optional[str] = None) -> Optional[str]:
    """
    Return content of a file inside the skill package.
    Default relative_path is SKILL.md. MLS may request references/, static/, manifest.yaml.
    """
    path = resolve_skill_path(skill_id, relative_path)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def list_skill_ids() -> List[str]:
    """Return list of available skill_id values."""
    return [m["skill_id"] for m in get_skills_metadata()]


def build_read_skill_response(
    skill_id: str,
    relative_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared JSON envelope for Expert (LangChain) and Agent (MCP) read_skill."""
    available = list_skill_ids()
    if not skill_id or skill_id not in available:
        # Keep error payload smaller when hundreds of ids exist.
        preview = available[:40]
        more = len(available) - len(preview)
        avail_msg = preview + ([f"...(+{more} more)"] if more > 0 else [])
        return {
            "success": False,
            "error": f"Unknown skill_id. Available (sample): {avail_msg}",
            "available_ids": available,
        }
    content = get_skill_content(skill_id, relative_path)
    rel = relative_path or "SKILL.md"
    if content is None:
        return {
            "success": False,
            "error": f"Could not read skill file: {skill_id}/{rel}",
            "skill_id": skill_id,
            "skill_root": get_skill_root(skill_id),
            "relative_path": rel,
            "available_ids": available,
        }
    meta = next((m for m in get_skills_metadata() if m["skill_id"] == skill_id), {})
    return {
        "success": True,
        "skill_id": skill_id,
        "skill_root": get_skill_root(skill_id),
        "relative_path": rel,
        "content": content,
        "source": meta.get("source", "venusfactory"),
        "available_ids": available,
    }


def build_list_skills_response() -> Dict[str, Any]:
    """Shared JSON envelope for listing skills (Agent MCP list_skills)."""
    return {
        "success": True,
        "skills": get_skills_metadata_string(),
        "available_ids": list_skill_ids(),
    }


def invalidate_skills_cache() -> None:
    """Clear cached skills metadata. Call after adding/removing skills at runtime."""
    _skill_id_to_root.cache_clear()
    get_skills_metadata.cache_clear()
    get_skills_metadata_string.cache_clear()
    get_skills_catalog_for_agent.cache_clear()
