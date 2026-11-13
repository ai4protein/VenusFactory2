"""
Skills middleware: load SKILL.md metadata and content for CB/MLS.

Discovers two roots:
  1. VenusFactory2 native: ``src/agent/skills/``
  2. scientific-agent-skills (optional submodule):
     ``src/agent/skills/vendor/scientific-agent-skills/skills/``

On name collisions, VF2 keeps the bare ``skill_id``; the scientific package is
registered as ``sas_<dirname>`` so the full upstream library stays addressable.
"""
from __future__ import annotations

import functools
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VF_SKILLS_DIR = Path(__file__).resolve().parent / "skills"
_SAS_SKILLS_DIR = _REPO_ROOT / "src" / "agent" / "skills" / "vendor" / "scientific-agent-skills" / "skills"
# Back-compat alias used by older callers/tests.
from agent.rsi.engine.context import get_resources as _get_research_resources
_research_resources = _get_research_resources()
if _research_resources is not None:
    _VF_SKILLS_DIR = _research_resources.skills
    # Third-party skills must be explicitly copied into this version's root.
    _SAS_SKILLS_DIR = _research_resources.workspace / "disabled_external_skills"
_SKILLS_DIR = _VF_SKILLS_DIR

_SAS_PREFIX = "sas_"
_MAX_SKILL_BYTES = 1024 * 1024

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
            package = path.resolve()
            if (not package.is_relative_to(root) or package == root
                    or _contained_file(package, package / "SKILL.md") is None):
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
            mapping[skill_id] = package
            claimed.add(dirname)
            claimed.add(skill_id)
    return mapping


def _contained_file(root: Path, path: Path) -> Path | None:
    try:
        resolved = path.resolve(strict=True)
        return resolved if resolved.is_relative_to(root) and resolved.is_file() else None
    except (OSError, RuntimeError, ValueError):
        return None


def _package_root(skill_id: str) -> Path | None:
    if not isinstance(skill_id, str) or not skill_id or skill_id.startswith("_") or "/" in skill_id or "\\" in skill_id:
        return None
    root = _skill_id_to_root().get(skill_id)
    if root is None:
        return None
    try:
        # Cached paths must not acquire a new symlink target after discovery.
        if root.resolve(strict=True) != root or not root.is_dir():
            return None
    except (OSError, RuntimeError):
        return None
    if not any(root != base and root.is_relative_to(base) for _, base in _skill_roots()):
        return None
    return root


def resolve_skill_path(skill_id: str, relative_path: Optional[str] = None) -> Optional[Path]:
    """Resolve only a package resource or an explicitly allowed shared fragment."""
    root = _package_root(skill_id)
    if root is None:
        return None
    if relative_path is None or relative_path in ("", "SKILL.md"):
        return _contained_file(root, root / "SKILL.md")
    if not isinstance(relative_path, str) or "\0" in relative_path:
        return None
    raw = relative_path.replace("\\", "/")
    if Path(raw).is_absolute() or PureWindowsPath(raw).drive:
        return None
    while raw.startswith("./"):
        raw = raw[2:]
    vf_root = _VF_SKILLS_DIR.resolve()
    if root.parent == vf_root:
        for prefix, _ in _SHARED_PATH_ALIASES:
            if not raw.startswith(prefix):
                continue
            suffix = Path(raw[len(prefix):])
            if suffix.is_absolute() or ".." in suffix.parts:
                return None
            shared_root = (vf_root / "_shared_nature").resolve()
            if shared_root == vf_root or not shared_root.is_relative_to(vf_root):
                return None
            return _contained_file(shared_root, shared_root / suffix)
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    return _contained_file(root, root / relative)


def _read_skill_file(path: Path) -> str | None:
    # Anchor the actual read to a configured root. A package/resource swapped
    # for a symlink after resolution is rejected by openat(O_NOFOLLOW), not read.
    from utils.scoped_files import read_bytes

    for _, root in _skill_roots():
        if path.is_relative_to(root):
            try:
                return read_bytes(root, path.relative_to(root).as_posix(), max_bytes=_MAX_SKILL_BYTES).decode("utf-8")
            except (OSError, ValueError, RuntimeError, UnicodeError):
                return None
    return None


def get_skill_root(skill_id: str) -> Optional[str]:
    """Return the authorized package root for host-side resource loading only."""
    root = _package_root(skill_id)
    if root is not None and _contained_file(root, root / "SKILL.md") is not None:
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
        path = resolve_skill_path(skill_id)
        raw = _read_skill_file(path) if path is not None else None
        if raw is None:
            continue
        meta = _parse_frontmatter(raw)
        nested = meta.get("metadata") if isinstance(meta.get("metadata"), dict) else {}
        name = meta.get("name", skill_id)
        source = root_source.get(str(pkg.parent.resolve()), "venusfactory")
        # pkg.parent is skills root; for VF2 pkg is under _VF_SKILLS_DIR
        if _SAS_SKILLS_DIR.is_dir() and pkg.resolve().is_relative_to(_SAS_SKILLS_DIR.resolve()):
            source = "scientific"
        elif pkg.parent.resolve() == _VF_SKILLS_DIR.resolve():
            source = "venusfactory"
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
    """Discover skills on demand instead of embedding the reference library."""
    return ("Skills are reference workflows, not proof that a tool is executable. "
            "Search list_skills(query=task, limit=10), then read_skill(skill_id) for the selected guide. "
            "Use relative_path only for a needed reference. Check live tool availability separately.")


def get_skill_content(skill_id: str, relative_path: Optional[str] = None) -> Optional[str]:
    """
    Return content of a file inside the skill package.
    Default relative_path is SKILL.md. MLS may request references/, static/, manifest.yaml.
    """
    path = resolve_skill_path(skill_id, relative_path)
    if path is None:
        return None
    return _read_skill_file(path)


def list_skill_ids() -> List[str]:
    """Return list of available skill_id values."""
    return [m["skill_id"] for m in get_skills_metadata()]


def build_read_skill_response(
    skill_id: str,
    relative_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Shared JSON envelope for Expert (LangChain) and Agent (MCP) read_skill."""
    root = _package_root(skill_id)
    if root is None:
        return {
            "success": False,
            "error": "Unknown or unavailable skill_id.",
            "next_step": "Search list_skills(query=task, limit=10), then read the selected skill_id.",
        }
    content = get_skill_content(skill_id, relative_path)
    rel = relative_path or "SKILL.md"
    if content is None:
        return {
            "success": False,
            "error": "Could not read the requested skill resource.",
            "skill_id": skill_id,
            "relative_path": rel,
            "next_step": "Use a relative resource path from the selected guide, or search list_skills.",
        }
    sas = _scientific_skills_dir()
    return {
        "success": True,
        "skill_id": skill_id,
        "relative_path": rel,
        "content": content,
        "source": "scientific" if sas is not None and root.is_relative_to(sas) else "venusfactory",
    }


def build_list_skills_response(query: str = "", limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    """Bounded bilingual skill index; file contents are read only on selection."""
    from mcp_tools.tools.discovery.matching import query_groups, relevance
    if not isinstance(query, str) or len(query) > 200 or type(limit) is not int or not 1 <= limit <= 30 or type(offset) is not int or offset < 0:
        return {"success": False, "error": "Use query up to 200 characters, limit 1–30 and a nonnegative offset."}
    groups = query_groups(query)
    found = []
    for item in get_skills_metadata():
        name = item["skill_id"]
        description = str(item.get("description", ""))
        score = relevance(groups, name, name + " " + description)
        if score is not None:
            found.append((score, name, {"skill_id": name, "description": description[:180], "source": item.get("source", "venusfactory")}))
    found.sort(key=lambda row: (-row[0], row[1]))
    page = [row[2] for row in found[offset:offset+limit]]
    return {"success": True, "items": page, "skills": "\n".join(row["skill_id"] + ": " + row["description"] for row in page),
            "available_ids": [row["skill_id"] for row in page], "total": len(found),
            "next_offset": offset + limit if offset + limit < len(found) else None}


def invalidate_skills_cache() -> None:
    """Clear cached skills metadata. Call after adding/removing skills at runtime."""
    _skill_id_to_root.cache_clear()
    get_skills_metadata.cache_clear()
    get_skills_metadata_string.cache_clear()
    get_skills_catalog_for_agent.cache_clear()
