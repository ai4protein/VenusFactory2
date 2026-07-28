"""
Skills middleware: load SKILL.md metadata and content for CB/MLS.
CB sees skill names and descriptions; MLS can read full SKILL (and package files) via read_skill.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, List, Optional

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"

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


def resolve_skill_path(skill_id: str, relative_path: Optional[str] = None) -> Optional[Path]:
    """Resolve a path inside a skill package (or whitelisted shared roots).

    Blocks arbitrary ``..`` traversal. Allows ``../_shared/...`` and
    ``_shared_nature/...`` only when they resolve under ``skills/_shared_nature/``.
    """
    if not skill_id or skill_id.startswith("_") or "/" in skill_id or "\\" in skill_id:
        return None
    skills_root = _SKILLS_DIR.resolve()
    root = (_SKILLS_DIR / skill_id).resolve()
    if not root.is_dir() or not str(root).startswith(str(skills_root)):
        return None
    if relative_path is None or relative_path in ("", "SKILL.md"):
        path = root / "SKILL.md"
        return path if path.is_file() else None

    rel_str = relative_path.replace("\\", "/").lstrip("./")
    # Shared-nature whitelist (nature_writing / nature_polishing always_load)
    for prefix, mapped in _SHARED_PATH_ALIASES:
        if rel_str.startswith(prefix) or relative_path.replace("\\", "/").startswith(prefix):
            raw = relative_path.replace("\\", "/")
            for pfx, _ in _SHARED_PATH_ALIASES:
                if raw.startswith(pfx):
                    raw = mapped + raw[len(pfx):]
                    break
            shared = (skills_root / raw).resolve()
            shared_root = (skills_root / "_shared_nature").resolve()
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
    root = (_SKILLS_DIR / skill_id).resolve()
    if root.is_dir() and (root / "SKILL.md").is_file():
        return str(root)
    return None


@functools.lru_cache(maxsize=1)
def get_skills_metadata() -> List[Dict[str, Any]]:
    """
    Discover all SKILL.md under src/agent/skills/ and return list of metadata dicts.
    Each dict has: skill_id, name, description, path, version, name_matches_dir.
    Directories starting with `_` (shared resources) are skipped.
    """
    result = []
    if not _SKILLS_DIR.exists():
        return result
    for path in sorted(_SKILLS_DIR.iterdir()):
        if not path.is_dir() or path.name.startswith("_"):
            continue
        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            continue
        try:
            raw = skill_md.read_text(encoding="utf-8")
        except Exception:
            continue
        meta = _parse_frontmatter(raw)
        skill_id = path.name
        nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        name = meta.get("name", skill_id)
        result.append({
            "skill_id": skill_id,
            "name": name,
            "description": meta.get("description", ""),
            "path": str(skill_md.relative_to(_SKILLS_DIR.parent.parent)),
            "version": nested.get("version") or meta.get("version", ""),
            "license": meta.get("license", ""),
            "name_matches_dir": name == skill_id,
        })
    return result


def _format_skills_metadata_string(*, max_desc: int = 800) -> str:
    items = get_skills_metadata()
    if not items:
        return "(No skills loaded.)"
    lines = []
    for s in items:
        # Bold skill_id so agents always use the directory name, not a display alias.
        sid = s.get("skill_id", "")
        desc = (s.get("description") or "")
        if max_desc > 0 and len(desc) > max_desc:
            desc = desc[: max_desc - 3] + "..."
        ver = s.get("version") or ""
        ver_bit = f" v{ver}" if ver else ""
        lines.append(f"- **{sid}**{ver_bit} (skill_id: `{sid}`): {desc}")
    return "\n".join(lines)


@functools.lru_cache(maxsize=1)
def get_skills_metadata_string() -> str:
    """Format skills metadata for Expert CB/MLS prompts (longer descriptions)."""
    return _format_skills_metadata_string(max_desc=800)


@functools.lru_cache(maxsize=1)
def get_skills_catalog_for_agent() -> str:
    """Compact catalog for Science Agent system prompt (self-directed loading)."""
    return _format_skills_metadata_string(max_desc=220)


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
    """Return list of available skill_id values (directory names)."""
    return [m["skill_id"] for m in get_skills_metadata()]


def build_read_skill_response(
    skill_id: str,
    relative_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared JSON envelope for Expert (LangChain) and Agent (MCP) read_skill."""
    available = list_skill_ids()
    if not skill_id or skill_id not in available:
        return {
            "success": False,
            "error": f"Unknown skill_id. Available: {available}",
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
    return {
        "success": True,
        "skill_id": skill_id,
        "skill_root": get_skill_root(skill_id),
        "relative_path": rel,
        "content": content,
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
    get_skills_metadata.cache_clear()
    get_skills_metadata_string.cache_clear()
    get_skills_catalog_for_agent.cache_clear()
