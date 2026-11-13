"""Skill resource reads stay inside explicit roots, including at open time."""
import json
import os

import pytest

from agent import skills


@pytest.fixture
def registry(tmp_path, monkeypatch):
    native = tmp_path / "native"
    vendor = tmp_path / "external-vendor" / "skills"
    package = native / "nature_writing"
    package.mkdir(parents=True)
    (package / "SKILL.md").write_text("---\nname: nature_writing\ndescription: Synthetic fixture\n---\nGUIDE")
    (package / "references").mkdir()
    (package / "references/guide.md").write_text("LOCAL_REFERENCE")
    shared = native / "_shared_nature"
    shared.mkdir()
    (shared / "valid.md").write_text("VALID_SHARED")
    sibling = native / "_shared_nature_extra"
    sibling.mkdir()
    (sibling / "marker.md").write_text("PREFIX_SIBLING_CANARY")
    external = tmp_path / "outside"
    external.mkdir()
    (external / "SKILL.md").write_text("OUTSIDE_PACKAGE_CANARY")
    (external / "marker.md").write_text("OUTSIDE_RESOURCE_CANARY")
    scientific = vendor / "reference_skill"
    scientific.mkdir(parents=True)
    (scientific / "SKILL.md").write_text("---\nname: reference_skill\ndescription: Vendor fixture\n---\nVENDOR_GUIDE")
    (scientific / "details.md").write_text("VENDOR_REFERENCE")
    monkeypatch.setattr(skills, "_VF_SKILLS_DIR", native)
    monkeypatch.setattr(skills, "_SAS_SKILLS_DIR", vendor)
    skills.invalidate_skills_cache()
    yield {"root": tmp_path, "native": native, "package": package, "shared": shared, "external": external, "vendor": vendor}
    skills.invalidate_skills_cache()


def assert_denied(skill_id, relative_path=None):
    from mcp_tools.tools.skill.tools_agent import read_skill_tool
    from mcp_tools.tools.skill.tools_mcp import mcp_read_skill

    direct = skills.build_read_skill_response(skill_id, relative_path)
    graph = json.loads(read_skill_tool.invoke({"skill_id": skill_id, "relative_path": relative_path}))
    native = json.loads(mcp_read_skill(skill_id, relative_path))
    assert direct == graph == native
    assert direct["success"] is False
    assert "content" not in direct and "skill_root" not in direct and "available_ids" not in direct
    assert "CANARY" not in json.dumps(direct)


@pytest.mark.parametrize("path", [
    "../_shared/../_shared_nature_extra/marker.md",
    "../_shared_nature/../_shared_nature_extra/marker.md",
    "_shared_nature/../_shared_nature_extra/marker.md",
    "../_shared/../../outside/marker.md",
    "../_shared/../nature_writing/SKILL.md",
    "../other/SKILL.md", "../../outside/marker.md", "..", "../SKILL.md",
    "/references/guide.md", "C:/references/guide.md", "C:references/guide.md",
])
def test_paths_cannot_cross_package_or_shared_boundary(registry, path):
    assert_denied("nature_writing", path)


@pytest.mark.parametrize("path", ["../_shared/valid.md", "../_shared_nature/valid.md", "_shared_nature/valid.md", "./_shared_nature/valid.md"])
def test_legacy_shared_aliases_remain_available(registry, path):
    assert skills.get_skill_content("nature_writing", path) == "VALID_SHARED"


@pytest.mark.parametrize("link", ["package", "manifest", "resource", "shared_root"])
def test_external_symlinks_never_load_skills_or_resources(registry, link):
    native, package, external = registry["native"], registry["package"], registry["external"]
    if link == "package":
        (native / "external_package").symlink_to(external, target_is_directory=True)
        skills.invalidate_skills_cache()
        assert_denied("external_package")
        assert "external_package" not in skills.list_skill_ids()
    elif link == "manifest":
        (package / "SKILL.md").unlink()
        (package / "SKILL.md").symlink_to(external / "marker.md")
        assert_denied("nature_writing")
        assert skills.get_skill_root("nature_writing") is None
    elif link == "resource":
        (package / "leak.md").symlink_to(external / "marker.md")
        assert_denied("nature_writing", "leak.md")
    else:
        registry["shared"].rename(native / "original_shared")
        registry["shared"].symlink_to(external, target_is_directory=True)
        assert_denied("nature_writing", "../_shared/marker.md")


def test_cached_package_cannot_acquire_external_symlink_target(registry):
    assert skills.get_skill_root("nature_writing") == str(registry["package"])
    registry["package"].rename(registry["native"] / "old_package")
    registry["package"].symlink_to(registry["external"], target_is_directory=True)
    assert_denied("nature_writing")


@pytest.mark.parametrize("replace", ["file_symlink", "directory_symlink", "fifo", "hardlink"])
def test_resource_swap_after_resolution_is_rejected_at_open(registry, monkeypatch, replace):
    original = skills.resolve_skill_path
    target = registry["package"] / "references/guide.md"
    changed = False

    def resolve_then_swap(skill_id, relative_path=None):
        nonlocal changed
        result = original(skill_id, relative_path)
        if relative_path == "references/guide.md" and not changed:
            assert result == target
            changed = True
            if replace == "directory_symlink":
                target.parent.rename(registry["package"] / "original_references")
                target.parent.symlink_to(registry["external"], target_is_directory=True)
            else:
                target.unlink()
                if replace == "file_symlink":
                    target.symlink_to(registry["external"] / "marker.md")
                elif replace == "fifo":
                    os.mkfifo(target)
                else:
                    os.link(registry["external"] / "marker.md", target)
        return result

    monkeypatch.setattr(skills, "resolve_skill_path", resolve_then_swap)
    assert skills.get_skill_content("nature_writing", "references/guide.md") is None
    assert changed


def test_explicit_external_root_and_relative_resources_are_supported(registry, monkeypatch):
    # The root itself is trusted operator configuration and may be outside the repository.
    alias = registry["root"] / "configured-root-link"
    alias.symlink_to(registry["native"], target_is_directory=True)
    monkeypatch.setattr(skills, "_VF_SKILLS_DIR", alias)
    skills.invalidate_skills_cache()
    assert skills.get_skill_root("nature_writing") == str(registry["package"])
    assert skills.get_skill_content("nature_writing", "./references/guide.md") == "LOCAL_REFERENCE"
    assert skills.get_skill_content("nature_writing", "references\\guide.md") == "LOCAL_REFERENCE"
    vendor = skills.build_read_skill_response("reference_skill", "details.md")
    assert vendor["success"] and vendor["source"] == "scientific" and vendor["content"] == "VENDOR_REFERENCE"
    assert skills.resolve_skill_path("reference_skill", "../_shared/valid.md") is None


def test_public_envelopes_do_not_export_host_paths_or_catalogue(registry):
    for i in range(50):
        package = registry["native"] / f"unrelated_{i}"
        package.mkdir()
        (package / "SKILL.md").write_text("---\nname: unrelated\ndescription: fixture\n---\nGUIDE")
    skills.invalidate_skills_cache()
    for value in [skills.build_read_skill_response("nature_writing"),
                  skills.build_read_skill_response("nature_writing", "missing.md"),
                  skills.build_read_skill_response("unknown")]:
        text = json.dumps(value)
        assert str(registry["root"]) not in text and "unrelated_" not in text
        assert "available_ids" not in value and "skill_root" not in value
    page = skills.build_list_skills_response(query="unrelated", limit=7)
    assert len(page["available_ids"]) == 7 and page["total"] == 50 and page["next_offset"] == 7


def test_skill_reads_are_bounded(registry):
    (registry["package"] / "large.md").write_bytes(b"x" * (skills._MAX_SKILL_BYTES + 1))
    assert_denied("nature_writing", "large.md")
