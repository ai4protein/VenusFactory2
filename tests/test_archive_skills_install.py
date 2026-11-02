"""Exercise source-release installation using local Git, never external services."""
import subprocess
from pathlib import Path

import pytest

from scripts.bootstrap.skills import install, PATH


def command(args):
    subprocess.run([str(value) for value in args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def upstream(tmp_path):
    repo = tmp_path / 'upstream'; repo.mkdir()
    command(['git', 'init', repo])
    (repo / 'skills/example').mkdir(parents=True)
    (repo / 'skills/example/SKILL.md').write_text('# Public reference')
    (repo / 'LICENSE').write_text('Test fixture')
    command(['git', '-C', repo, 'add', '.'])
    command(['git', '-C', repo, '-c', 'user.name=Test Fixture', '-c', 'user.email=fixture@example.org', 'commit', '-m', 'Fixture'])
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    return {'repository': repo.as_uri(), 'revision': revision}


def test_archive_installs_exact_revision_and_reuses_it(tmp_path):
    metadata = upstream(tmp_path)
    root = tmp_path / 'source'; root.mkdir()
    assert install(root, command, metadata)['revision'] == metadata['revision']
    assert (root / PATH / 'LICENSE').is_file()
    assert install(root, command, metadata)['status'] == 'ready'
    (root / PATH / 'skills/example/SKILL.md').write_text('Local work')
    with pytest.raises(ValueError, match='local modifications'):
        install(root, command, metadata)
    assert (root / PATH / 'skills/example/SKILL.md').read_text() == 'Local work'


def test_existing_non_git_directory_is_not_overwritten(tmp_path):
    metadata = upstream(tmp_path)
    root = tmp_path / 'source'; (root / PATH).mkdir(parents=True)
    (root / PATH / 'notes.txt').write_text('User work')
    with pytest.raises(ValueError, match='not empty'):
        install(root, command, metadata)
    assert (root / PATH / 'notes.txt').read_text() == 'User work'
