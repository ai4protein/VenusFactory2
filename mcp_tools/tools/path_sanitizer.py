from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def to_project_relative_path(path: str | Path) -> str:
    """Return project-relative path when possible, otherwise basename."""
    target = Path(path).expanduser()
    if not target.is_absolute():
        return target.as_posix()

    resolved = target.resolve()
    try:
        return resolved.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.name


def to_client_file_path(path: str | Path) -> str:
    """Convert file path to user-safe project-relative form."""
    target = Path(path).expanduser()
    if target.exists():
        return to_project_relative_path(target.resolve())
    return to_project_relative_path(target)
