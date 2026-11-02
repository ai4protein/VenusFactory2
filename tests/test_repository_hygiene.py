"""Keep deployment state and private material out of source-control candidates."""
import importlib.util
import io
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repository_hygiene', ROOT / 'scripts/check_repository.py')
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_runtime_and_credentials_are_not_source_files():
    for path in ['.env', '.env.online', 'runtime/local/sessions/a.json', 'configs/model_runtimes.json', 'configs/rsi/paper_steps.json', 'ckpt/model.pt']:
        assert guard.private_path(path)
    for path in ['.env.example', 'configs/model_runtimes.example.json', 'ckpt/manifest.json', 'src/agent/harness/gateway.py']:
        assert not guard.private_path(path)


def test_credentials_are_reported_without_echoing_the_value():
    secret = b'sk-or-v1-' + b'a' * 64
    findings = guard.inspect_bytes('settings.py', secret)
    assert findings and secret.decode() not in '\n'.join(findings)


def test_safe_zip_and_private_zip_have_different_outcomes():
    def bundle(name, text):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            archive.writestr(name, text)
        return data.getvalue()
    assert not guard.inspect_bytes('example.zip', bundle('README.md', 'Public example'))
    assert guard.inspect_bytes('example.zip', bundle('.env', 'private'))
    assert guard.inspect_bytes('example.zip', bundle('../outside.txt', 'escape'))
    assert guard.inspect_bytes('example.zip', bundle('config.txt', b'olp_' + b'a' * 32))


def test_corrupt_archives_fail_closed():
    assert guard.inspect_bytes('example.zip', b'not a zip')
