"""Wire VenusFactory + scientific-agent-skills into kimi-code Skill discovery.

Kimi discovers project skills under ``<git-root>/.kimi-code/skills/<name>/``
(and ``.agents/skills/``). VenusFactory authoring lives in
``src/agent/skills/``; the full scientific-agent-skills library (optional
submodule) is exposed via the dual-root loader in ``agent.skills``.

This module keeps a project-level symlink tree so Science Agent (local) can
also use the built-in ``Skill`` tool. Symlink names use the public
``skill_id`` (including ``sas_<id>`` on collisions).

Online Science Agent should prefer MCP ``mcp__venusfactory__read_skill``
because the sandbox may not see the repo tree and ``Skill`` is security-denied.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_KIMI_SKILLS = _REPO_ROOT / ".kimi-code" / "skills"


def ensure_kimi_project_skills(repo_root: Path | None = None) -> Path:
    """Ensure ``.kimi-code/skills/<skill_id>`` symlinks for VF2 + scientific skills.

    Idempotent. Uses ``agent.skills._skill_id_to_root()`` so collision IDs
    (``sas_*``) match ``read_skill``. Returns the kimi skills directory path.
    """
    root = Path(repo_root) if repo_root else _REPO_ROOT
    dest = root / ".kimi-code" / "skills"
    dest.mkdir(parents=True, exist_ok=True)

    try:
        # Import inside so missing/broken submodule only affects discovery.
        from agent.skills import invalidate_skills_cache, _skill_id_to_root

        invalidate_skills_cache()
        id_map = _skill_id_to_root()
    except Exception as exc:
        _logger.warning("skill map unavailable for kimi links: %s", exc)
        return dest

    wanted: set[str] = set()
    for skill_id, child in id_map.items():
        if not child.is_dir() or not (child / "SKILL.md").is_file():
            continue
        wanted.add(skill_id)
        link = dest / skill_id
        target = os.path.relpath(child.resolve(), start=dest.resolve())
        try:
            if link.is_symlink() or link.exists():
                if link.is_symlink() and os.path.realpath(link) == str(child.resolve()):
                    continue
                if link.is_symlink() or link.is_file():
                    link.unlink()
                elif link.is_dir() and not link.is_symlink():
                    _logger.warning("skip skill link %s: real directory exists", link)
                    continue
            link.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            _logger.warning("failed to link skill %s: %s", skill_id, exc)

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
