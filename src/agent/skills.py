"""
Skills middleware: load SKILL.md metadata and content for CB/MLS.
CB sees skill names and descriptions; MLS can read full SKILL via read_skill tool.
"""
import functools
from pathlib import Path
from typing import Dict, List, Any, Optional

_SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def _parse_frontmatter(text: str) -> Dict[str, Any]:
    """Parse YAML-like frontmatter between first --- and second ---.

    Handles three forms:
      key: value                       — single-line scalar
      key: >-                          — YAML folded scalar (lines joined with spaces, no final newline)
        line1
        line2
      key: >                           — YAML folded scalar (lines joined with spaces, trailing newline)
        line1
    """
    out: Dict[str, Any] = {}
    if not text.strip().startswith("---"):
        return out
    parts = text.split("---", 2)
    if len(parts) < 3:
        return out
    lines = parts[1].strip().splitlines()

    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        if ":" not in stripped:
            i += 1
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        value = value.strip()
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
                # Detect indentation
                indent = len(cont) - len(cont.lstrip())
                if base_indent is None:
                    base_indent = indent
                if indent < base_indent or indent == 0:
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
    return out


@functools.lru_cache(maxsize=1)
def get_skills_metadata() -> List[Dict[str, Any]]:
    """
    Discover all SKILL.md under src/agent/skills/ and return list of metadata dicts.
    Each dict has: skill_id (dir name), name, description, path (relative).
    """
    result = []
    if not _SKILLS_DIR.exists():
        return result
    for path in sorted(_SKILLS_DIR.iterdir()):
        if not path.is_dir():
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
        result.append({
            "skill_id": skill_id,
            "name": meta.get("name", skill_id),
            "description": meta.get("description", ""),
            "path": str(skill_md.relative_to(_SKILLS_DIR.parent.parent)),
        })
    return result


@functools.lru_cache(maxsize=1)
def get_skills_metadata_string() -> str:
    """Format skills metadata for injection into CB prompt (one line per skill)."""
    items = get_skills_metadata()
    if not items:
        return "(No skills loaded.)"
    lines = []
    for s in items:
        name = s.get("name", s.get("skill_id", ""))
        desc = (s.get("description") or "")[:200]
        lines.append(f"- **{name}** (skill_id: `{s['skill_id']}`): {desc}")
    return "\n".join(lines)


def get_skill_content(skill_id: str) -> Optional[str]:
    """
    Return full content of SKILL.md for the given skill_id (directory name).
    MLS can use this to read and execute instructions from the skill.
    """
    path = _SKILLS_DIR / skill_id / "SKILL.md"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def list_skill_ids() -> List[str]:
    """Return list of available skill_id values (directory names)."""
    return [m["skill_id"] for m in get_skills_metadata()]


def invalidate_skills_cache() -> None:
    """Clear cached skills metadata. Call after adding/removing skills at runtime."""
    get_skills_metadata.cache_clear()
    get_skills_metadata_string.cache_clear()
