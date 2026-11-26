from pathlib import Path
import pytest
from backend.web_v2.runtime_paths import AREAS, configure_runtime_paths, runtime_root


def test_modes_use_disjoint_roots_for_all_mutable_services(tmp_path):
    modes = []
    for mode in ('local', 'online'):
        env = {'WEBUI_V2_MODE': mode, 'VENUS_RUNTIME_ROOT': str(tmp_path)}
        root = configure_runtime_paths(env=env)
        assert root == tmp_path / mode
        assert all(Path(env[key]).is_relative_to(root) for key in AREAS)
        modes.append(set(env[key] for key in AREAS))
    assert not modes[0].intersection(modes[1])


def test_explicit_legacy_paths_are_preserved_without_silent_migration(tmp_path):
    old = str(tmp_path / 'existing.sqlite3')
    env = {'VENUSFACTORY_SESSION_DB_PATH': old, 'TEMP_OUTPUTS_DIR': '  '}
    root = configure_runtime_paths(env=env)
    assert env['VENUSFACTORY_SESSION_DB_PATH'] == old
    assert env['TEMP_OUTPUTS_DIR'] == str(root / 'files')


def test_unknown_mode_cannot_alias_a_valid_store():
    with pytest.raises(ValueError): runtime_root('../online')
