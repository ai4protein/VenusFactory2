import json
from pathlib import Path

import pytest

from mcp_tools.models.profiles import load_runtime_settings, repository_path


def test_paths_survive_worker_chdir(tmp_path, monkeypatch):
    repo = tmp_path / 'repository with spaces'
    (repo / 'configs').mkdir(parents=True)
    profile = {'ligandmpnn': {'enabled': True, 'python': 'runtime/local/models/env/bin/python',
                             'weights': 'runtime/local/models/weights.pt', 'timeout_seconds': 10}}
    (repo / 'configs/model_runtimes.json').write_text(json.dumps(profile))
    monkeypatch.chdir(tmp_path)
    cfg = load_runtime_settings(root=repo, env={})['ligandmpnn']
    assert cfg['python'] == str(repo / 'runtime/local/models/env/bin/python')
    assert cfg['weights'] == str(repo / 'runtime/local/models/weights.pt')
    assert cfg['timeout_seconds'] == 10
    assert cfg['enabled'] is True


def test_override_and_external_installation(tmp_path, monkeypatch):
    external = tmp_path / 'external/bin/vina'
    (tmp_path / 'alternate.json').write_text(json.dumps({'vina': {'enabled': False, 'executable': str(external)}}))
    monkeypatch.chdir(tmp_path.parent)
    cfg = load_runtime_settings(root=tmp_path, env={'VENUS_MODEL_RUNTIMES_CONFIG': 'alternate.json'})
    assert cfg['vina']['executable'] == str(external)
    assert cfg['vina']['enabled'] is False
    assert repository_path('configs', root=tmp_path) == tmp_path / 'configs'


@pytest.mark.parametrize('profile', [[], {'vina': []}, {'vina': {'enabled': 'false'}}, {'vina': {'python': 12}}])
def test_reject_malformed_profiles(tmp_path, profile):
    p = tmp_path / 'profile.json'
    p.write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        load_runtime_settings(root=tmp_path, env={'VENUS_MODEL_RUNTIMES_CONFIG': str(p)})


def test_missing_installation_does_not_enable_models(tmp_path):
    assert load_runtime_settings(root=tmp_path, env={}) == {}


def test_virtual_environment_python_symlink_is_preserved(tmp_path):
    base = tmp_path / 'base-python'
    base.write_text('fixture executable')
    binary = tmp_path / 'runtime/env/bin/python'
    binary.parent.mkdir(parents=True)
    binary.symlink_to(base)
    assert repository_path('runtime/env/bin/python', root=tmp_path) == binary
    assert repository_path('runtime/env/bin/python', root=tmp_path) != base
