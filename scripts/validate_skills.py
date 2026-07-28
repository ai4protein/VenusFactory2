#!/usr/bin/env python3
"""Validate VenusFactory2 agent skills (Agent Skills–inspired VF2 profile).

Checks (VF2 tree — strict):
  - Each skill dir (non-_*) has SKILL.md
  - Frontmatter has name + description
  - name == directory name (skill_id)
  - Top-level keys ⊆ Agent Skills closed set (+ legacy version/author warned)
  - metadata.version present (warn if missing)
  - SKILL.md line count soft-cap (default 500)

Checks (scientific-agent-skills submodule — loose):
  - If ``third_party/scientific-agent-skills/skills`` exists, each package
    dir must contain SKILL.md. Frontmatter differences do not fail CI.

Usage:
  python3 scripts/validate_skills.py
  python3 scripts/validate_skills.py --strict
  python3 scripts/validate_skills.py --check-refs
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / "src" / "agent" / "skills"
SAS_SKILLS_DIR = ROOT / "third_party" / "scientific-agent-skills" / "skills"
ALLOWED_TOP = {
    "name",
    "description",
    "license",
    "compatibility",
    "allowed-tools",
    "metadata",
}
# Legacy nature_* used top-level version/author; warn, fail only in --strict after migration.
LEGACY_TOP = {"version", "author"}
SOFT_LINE_LIMIT = 500
# Only match concrete paths (skip placeholders like references/<section>.md)
_REF_PATH_RE = re.compile(
    r"(?:relative_path\s*=\s*[\"']([^\"'<>]+)[\"']|"
    r"`((?:references|static)/[^`\s<>)]+\.md)`|"
    r"`(manifest\.yaml)`|"
    r"`(_shared_nature/[^`\s<>)]+\.md)`)"
)


def _parse_frontmatter(text: str) -> dict[str, str]:
    if not text.strip().startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    out: dict[str, str] = {}
    in_metadata = False
    lines = parts[1].splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        if in_metadata and indent > 0 and ":" in stripped:
            k, _, v = stripped.partition(":")
            out[f"metadata.{k.strip()}"] = v.strip().strip("'\"")
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
        if key == "metadata":
            in_metadata = True
            i += 1
            continue
        if value in (">", ">-", "|", "|-"):
            out[key] = "<folded>"
            j = i + 1
            while j < len(lines):
                cont = lines[j]
                if not cont.strip():
                    j += 1
                    continue
                cont_indent = len(cont) - len(cont.lstrip(" "))
                if cont_indent == 0:
                    break
                j += 1
            i = j
            continue
        out[key] = value.strip("'\"")
        i += 1
    return out


def _check_referenced_paths(skill_id: str, body: str, check_refs: bool) -> list[str]:
    """Return warnings/errors for broken relative_path / references links."""
    if not check_refs:
        return []
    msgs: list[str] = []
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from agent.skills import resolve_skill_path
    except Exception as e:
        return [f"{skill_id}: --check-refs skipped (import): {e}"]

    seen: set[str] = set()
    for m in _REF_PATH_RE.finditer(body):
        rel = next((g for g in m.groups() if g), None)
        if not rel or rel in seen:
            continue
        seen.add(rel)
        # skip external URLs mistaken as paths
        if rel.startswith("http"):
            continue
        if resolve_skill_path(skill_id, rel) is None:
            msgs.append(f"{skill_id}: broken ref path `{rel}`")
    # nature manifests: only validate the always_load list (not axis value maps)
    manifest = SKILLS_DIR / skill_id / "manifest.yaml"
    if manifest.is_file():
        in_always = False
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            if re.match(r"^always_load:\s*$", raw.strip()):
                in_always = True
                continue
            if in_always:
                if raw and not raw.startswith(" ") and not raw.startswith("\t") and raw.strip():
                    in_always = False
                    continue
                m = re.match(r"^\s*-\s+(\S+\.md)\s*$", raw)
                if not m:
                    continue
                rel = m.group(1)
                if resolve_skill_path(skill_id, rel) is None:
                    msgs.append(f"{skill_id}: manifest always_load missing `{rel}`")
    return msgs


def validate(strict: bool = False, check_refs: bool = False) -> int:
    if not SKILLS_DIR.is_dir():
        print(f"ERROR: skills dir missing: {SKILLS_DIR}", file=sys.stderr)
        return 2

    errors: list[str] = []
    warnings: list[str] = []
    skill_count = 0

    for path in sorted(SKILLS_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("_"):
            if (path / "SKILL.md").exists():
                errors.append(f"{path.name}: shared dirs must not contain SKILL.md")
            continue
        skill_md = path / "SKILL.md"
        if not skill_md.exists():
            errors.append(f"{path.name}: missing SKILL.md")
            continue
        skill_count += 1
        text = skill_md.read_text(encoding="utf-8")
        meta = _parse_frontmatter(text)
        if not meta.get("name"):
            errors.append(f"{path.name}: missing frontmatter name")
        elif meta["name"] != path.name and meta["name"] != "<folded>":
            errors.append(
                f"{path.name}: name '{meta['name']}' != directory (skill_id must match)"
            )
        if not meta.get("description"):
            errors.append(f"{path.name}: missing description")
        top_keys = {k for k in meta if not k.startswith("metadata.")}
        unknown = top_keys - ALLOWED_TOP - LEGACY_TOP
        if unknown:
            msg = f"{path.name}: unknown top-level keys {sorted(unknown)}"
            (errors if strict else warnings).append(msg)
        legacy = top_keys & LEGACY_TOP
        if legacy:
            warnings.append(
                f"{path.name}: legacy top-level {sorted(legacy)} — move into metadata"
            )
        if not meta.get("metadata.version") and "version" not in meta:
            warnings.append(f"{path.name}: missing metadata.version")
        desc = meta.get("description") or ""
        if desc and desc != "<folded>":
            if not re.search(r"(?i)\buse when\b|\buse whenever\b|\buse for\b|\buse to\b", desc):
                warnings.append(f"{path.name}: description missing Use when/for trigger phrasing")
            if not re.search(r"(?i)\bdo not use\b|\bdon't use\b|\bnot for\b|\bprefer\b", desc):
                warnings.append(f"{path.name}: description missing Do NOT use / Not for / Prefer routing")
        n_lines = text.count("\n") + 1
        if n_lines > SOFT_LINE_LIMIT:
            warnings.append(
                f"{path.name}: SKILL.md has {n_lines} lines (soft limit {SOFT_LINE_LIMIT}); "
                "prefer references/ progressive disclosure"
            )
        body = text.split("---", 2)[2] if text.count("---") >= 2 else text
        for msg in _check_referenced_paths(path.name, body, check_refs):
            (errors if strict else warnings).append(msg)

    # Loose scan of optional scientific-agent-skills submodule
    sas_count = 0
    if SAS_SKILLS_DIR.is_dir():
        for path in sorted(SAS_SKILLS_DIR.iterdir()):
            if not path.is_dir() or path.name.startswith("_"):
                continue
            sas_count += 1
            if not (path / "SKILL.md").is_file():
                errors.append(
                    f"scientific/{path.name}: missing SKILL.md "
                    "(loose submodule check)"
                )
    elif (ROOT / "third_party" / "scientific-agent-skills").exists():
        warnings.append(
            "scientific-agent-skills present but skills/ missing — "
            "run: git submodule update --init --recursive"
        )

    # Loader smoke (optional if import path works)
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from agent.skills import get_skills_metadata, invalidate_skills_cache

        invalidate_skills_cache()
        loaded = get_skills_metadata()
        vf_loaded = [m for m in loaded if m.get("source") == "venusfactory"]
        sas_loaded = [m for m in loaded if m.get("source") == "scientific"]
        if len(vf_loaded) != skill_count:
            errors.append(
                f"loader registered {len(vf_loaded)} venusfactory skills "
                f"but filesystem has {skill_count}"
            )
        if SAS_SKILLS_DIR.is_dir() and sas_count and len(sas_loaded) == 0:
            errors.append(
                "scientific skills dir present but loader registered 0 "
                "scientific skills"
            )
        for m in vf_loaded:
            if not m.get("name_matches_dir"):
                warnings.append(
                    f"{m['skill_id']}: loader reports name_matches_dir=False "
                    f"(name={m.get('name')!r})"
                )
    except Exception as e:
        warnings.append(f"loader smoke skipped: {e}")

    for w in warnings:
        print(f"WARN: {w}")
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)

    print(
        f"Validated {skill_count} VF2 skills"
        + (f" + {sas_count} scientific (loose)" if sas_count else "")
        + f" — {len(errors)} error(s), {len(warnings)} warning(s)"
    )
    if errors:
        return 1
    if strict and warnings:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings (and unknown keys) as failures",
    )
    parser.add_argument(
        "--check-refs",
        action="store_true",
        help="Resolve relative_path / references / nature manifest paths via skills loader",
    )
    args = parser.parse_args()
    raise SystemExit(validate(strict=args.strict, check_refs=args.check_refs))


if __name__ == "__main__":
    main()
