"""Content identity shared by independently started service processes."""
from functools import lru_cache
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_revision(root=ROOT):
    """Bind executable source and shipped configuration, including uncommitted edits.

    No Git installation is required. Runtime data, credentials, model weights and
    bytecode are outside this inventory. Paths are relative so release copies have
    the same identity. This is a version check, not a live source-change watcher.
    """
    root = Path(root)
    files = list(root.glob("*.py"))
    for directory in ("backend", "mcp_tools", "src"):
        files.extend((root / directory).rglob("*.py"))
    for path in (root / "configs").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".toml", ".yaml", ".yml"}:
            files.append(path)
    digest = hashlib.sha256()
    for path in sorted(files):
        if "__pycache__" in path.parts:
            continue
        name = path.relative_to(root).as_posix().encode()
        digest.update(len(name).to_bytes(4, "big"))
        digest.update(name)
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def loaded_source_revision():
    """Freeze the process's generation; a running server must not claim new code."""
    return source_revision()
