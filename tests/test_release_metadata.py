import json
import shutil
from pathlib import Path

from scripts.check_release import check


def test_release_metadata_is_consistent():
    assert check() == []


def test_mismatched_frontend_version_blocks_release(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in ['pyproject.toml', 'frontend/package.json', 'frontend/package-lock.json', 'docker-compose.yaml', 'docs/releases/v0.3.0.md']:
        target = tmp_path / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    p = tmp_path / 'frontend/package.json'; value = json.loads(p.read_text()); value['version'] = '0.0.0'; p.write_text(json.dumps(value))
    assert any('frontend' in issue for issue in check(tmp_path))
