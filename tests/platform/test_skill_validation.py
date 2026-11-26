"""Optional reference skills must not make a minimal checkout invalid."""
from scripts import validate_skills


def test_empty_optional_submodule_is_valid(tmp_path, monkeypatch, capsys):
    vendor = tmp_path / "scientific-agent-skills"
    vendor.mkdir()
    monkeypatch.setattr(validate_skills, "SAS_SKILLS_DIR", vendor / "skills")
    assert validate_skills.validate(strict=True) == 0
    assert "0 warning(s)" in capsys.readouterr().out


def test_incomplete_optional_install_remains_a_warning(tmp_path, monkeypatch, capsys):
    vendor = tmp_path / "scientific-agent-skills"
    vendor.mkdir()
    (vendor / "README.md").write_text("Incomplete download")
    monkeypatch.setattr(validate_skills, "SAS_SKILLS_DIR", vendor / "skills")
    assert validate_skills.validate(strict=True) == 1
    assert "installation is incomplete" in capsys.readouterr().out
