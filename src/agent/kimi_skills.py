"""Wire VenusFactory ``src/agent/skills`` into kimi-code's native Skill discovery.

Kimi discovers project skills under ``<git-root>/.kimi-code/skills/<name>/``
(and ``.agents/skills/``). VenusFactory authoring lives in
``src/agent/skills/`` for the Science Expert ``read_skill`` path. This module
keeps a project-level symlink tree so Science Agent (local) can also use the
built-in ``Skill`` tool on the same packages.

Online Science Agent should prefer MCP ``mcp__venusfactory__read_skill``
because the sandbox may not see the repo tree and ``Skill`` is security-denied.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VF_SKILLS = _REPO_ROOT / "src" / "agent" / "skills"
_KIMI_SKILLS = _REPO_ROOT / ".kimi-code" / "skills"


def ensure_kimi_project_skills(repo_root: Path | None = None) -> Path:
    """Ensure ``.kimi-code/skills/<skill_id>`` symlinks exist for each VF skill.

    Idempotent. Skips ``_*`` shared dirs and non-skill folders without SKILL.md.
    Returns the kimi skills directory path.
    """
    root = Path(repo_root) if repo_root else _REPO_ROOT
    src = root / "src" / "agent" / "skills"
    dest = root / ".kimi-code" / "skills"
    dest.mkdir(parents=True, exist_ok=True)

    if not src.is_dir():
        _logger.warning("VF skills dir missing: %s", src)
        return dest

    wanted: set[str] = set()
    for child in sorted(src.iterdir()):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        if not (child / "SKILL.md").is_file():
            continue
        wanted.add(child.name)
        link = dest / child.name
        target = os.path.relpath(child.resolve(), start=dest.resolve())
        try:
            if link.is_symlink() or link.exists():
                if link.is_symlink() and os.path.realpath(link) == str(child.resolve()):
                    continue
                if link.is_symlink() or link.is_file():
                    link.unlink()
                elif link.is_dir() and not link.is_symlink():
                    # Do not delete a real directory that a user may have populated.
                    _logger.warning("skip skill link %s: real directory exists", link)
                    continue
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            _logger.warning("failed to link skill %s: %s", child.name, exc)

    # Remove stale symlinks for deleted skills
    for link in dest.iterdir():
        if link.name in wanted:
            continue
        if link.is_symlink():
            try:
                link.unlink()
            except OSError:
                pass

    return dest


def kimi_skills_dir() -> Path:
    return _KIMI_SKILLS
