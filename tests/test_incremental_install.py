"""Real failure boundaries for incremental installation; no external calls."""
import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

from scripts.bootstrap import assets, cli, harnesses, progress, system
from scripts.bootstrap.common import Runner, atomic_json


MANIFEST = {'profiles': {'minimal': [], 'standard': ['proteinmpnn']}, 'tools': {'proteinmpnn': {}, 'ligandmpnn': {}}}


def test_default_does_not_select_models_or_full_platform():
    args = cli.parser().parse_args(['install'])
    assert args.profile == 'minimal'
    assert cli.selection(args, MANIFEST) == (['environment'], [])


@pytest.mark.parametrize('flags,stages,tools', [
    (['--tools', 'ligandmpnn'], ['tools'], ['ligandmpnn']),
    (['--harnesses', 'codex'], ['harnesses'], []),
    (['--factory-preset', 'demo'], ['factory'], []),
    (['--backbone', 'facebook/esm2_t6_8M_UR50D'], ['factory'], []),
    (['--harnesses', 'codex', '--tools', 'proteinmpnn'], ['harnesses', 'tools'], ['proteinmpnn']),
    (['--stages', 'frontend', '--tools', 'proteinmpnn'], ['frontend'], ['proteinmpnn']),
])
def test_component_flags_do_not_install_unrelated_components(flags, stages, tools):
    assert cli.selection(cli.parser().parse_args(['install', *flags]), MANIFEST) == (stages, tools)


def test_manager_reuses_available_manager(monkeypatch):
    monkeypatch.setattr(system, 'executable', lambda name: '/conda' if name == 'conda' else None)
    assert system.choose_manager() == 'conda'
    assert system.choose_manager('uv') == 'uv'
    monkeypatch.setattr(system, 'executable', lambda name: '/tool')
    assert system.choose_manager() == 'uv'


def test_log_progress_never_contains_terminal_controls():
    output = io.StringIO()
    ui = progress.Progress(output)
    ui.update('weights', 2, 50, 100, force=True)
    ui.line('done')
    assert '50%' in output.getvalue() and 'ETA' in output.getvalue()
    assert '\x1b' not in output.getvalue() and '\r' not in output.getvalue()


class Response(io.BytesIO):
    def __init__(self, value, status=200, **headers):
        super().__init__(value)
        self.status, self.headers = status, headers


def partial(path, data, expected):
    path.with_name(path.name + '.part').write_bytes(data)
    atomic_json(path.with_name(path.name + '.part.json'), {'url': 'https://example.org/file', 'expected': expected, 'algorithm': 'sha256'})


def test_interrupted_download_resumes_and_verifies(tmp_path, monkeypatch):
    path = tmp_path / 'weights'
    expected = hashlib.sha256(b'abcdef').hexdigest()
    partial(path, b'abc', expected)
    def request(req, **kw):
        assert req.headers['Range'] == 'bytes=3-'
        return Response(b'def', 206, **{'Content-Range': 'bytes 3-5/6'})
    monkeypatch.setattr(assets.urllib.request, 'urlopen', request)
    assets.download('https://example.org/file', path, expected=expected)
    assert path.read_bytes() == b'abcdef'
    assert not path.with_name('weights.part').exists()


def test_complete_partial_is_recovered_without_range_416(tmp_path, monkeypatch):
    path = tmp_path / 'weights'
    expected = hashlib.sha256(b'abcdef').hexdigest()
    partial(path, b'abcdef', expected)
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *a, **k: pytest.fail('No HTTP request needed'))
    assets.download('https://example.org/file', path, expected=expected)
    assert path.read_bytes() == b'abcdef'


def test_server_ignoring_range_restarts_without_duplicate_bytes(tmp_path, monkeypatch):
    path = tmp_path / 'weights'
    expected = hashlib.sha256(b'abcdef').hexdigest()
    partial(path, b'abc', expected)
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *a, **k: Response(b'abcdef', 200, **{'Content-Length': '6'}))
    assets.download('https://example.org/file', path, expected=expected)
    assert path.read_bytes() == b'abcdef'


def test_incomplete_response_is_never_promoted(tmp_path, monkeypatch):
    path = tmp_path / 'weights'
    expected = hashlib.sha256(b'abcdef').hexdigest()
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *a, **k: Response(b'abc', 200, **{'Content-Length': '6'}))
    with pytest.raises(OSError, match='Incomplete'):
        assets.download('https://example.org/file', path, expected=expected)
    assert not path.exists()
    assert path.with_name('weights.part').read_bytes() == b'abc'


def test_wrong_range_rejected_without_modifying_partial(tmp_path, monkeypatch):
    path = tmp_path / 'weights'
    expected = 'a' * 64
    partial(path, b'abc', expected)
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *a, **k: Response(b'def', 206, **{'Content-Range': 'bytes 0-2/6'}))
    with pytest.raises(ValueError, match='Invalid partial'):
        assets.download('https://example.org/file', path, expected=expected)
    assert path.with_name('weights.part').read_bytes() == b'abc'


def test_runner_timeout_kills_child_and_does_not_report_success(tmp_path):
    import subprocess
    with pytest.raises(subprocess.TimeoutExpired):
        Runner(tmp_path, tmp_path / 'log')([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=0.1)


def test_runner_explains_rate_limits_without_dumping_urls(tmp_path):
    with pytest.raises(RuntimeError, match='HTTP 429 download rate limit') as error:
        Runner(tmp_path, tmp_path / 'log')([sys.executable, '-c', 'print("429 Client Error: https://example.org/private?token=hidden"); raise SystemExit(1)'])
    assert 'token=' not in str(error.value) and 'hidden' not in str(error.value)


def test_harness_discovery_preserves_explicit_configuration(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'src/agent/harness/installation.py'
    spec = importlib.util.spec_from_file_location('bootstrap_discovery_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    binary = tmp_path / 'runtime/local/harnesses/codex/codex'
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b'binary')
    atomic_json(binary.parent / 'installation.json', {'paths': {'codex_bin': str(binary.relative_to(tmp_path))}})
    assert module.discover(tmp_path, {'python_bin': '/own/python'})['codex_bin'] == str(binary)
    monkeypatch.setenv('VENUS_CODEX_BIN', '/configured/codex')
    assert module.discover(tmp_path)['codex_bin'] == '/configured/codex'
    assert module.discover(tmp_path, {'python_bin': '/own/python'})['python_bin'] == '/own/python'


def test_harness_names_validate_before_mutation():
    assert harnesses.names('codex,codex') == ['codex']
    with pytest.raises(ValueError, match='Unknown harness'):
        harnesses.names('unverified-package')


def test_sdk_and_cli_prerelease_formats_refer_to_same_version():
    assert harnesses.version_matches('0.1.5-rc.1', '0.1.5rc1')
    assert harnesses.version_matches('codex-cli 0.163.0', '0.163.0')
    assert not harnesses.version_matches('0.163.01', '0.163.0')


def test_base_install_never_runs_platform_dependency_recipe(tmp_path, monkeypatch):
    atomic_json(tmp_path / 'configs/bootstrap.json', MANIFEST)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    monkeypatch.setattr(cli, 'probe', lambda root: {'os': 'Linux', 'arch': 'x86_64', 'free_gib': 2, 'gpu': False,
                                                'torch_type': 'cpu', 'executables': {'git': None, 'bwrap': None}})
    def create(self, folder, version):
        assert version == '3.12'
        return folder / 'bin/python'
    monkeypatch.setattr(system.Environment, 'create', create)
    monkeypatch.setattr(system.Environment, 'platform', lambda *a: pytest.fail('base install must not install ML packages'))
    monkeypatch.setattr(cli, 'doctor', lambda *a, **k: {'environment': {'status': 'ready'}})
    assert cli.main(['install', '--no-progress']) == 0


def test_json_plan_stdout_is_machine_readable(tmp_path, monkeypatch, capsys):
    atomic_json(tmp_path / 'configs/bootstrap.json', MANIFEST)
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    assert cli.main(['plan', '--harnesses', 'codex', '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['stages'] == ['harnesses'] and result['tools'] == []
