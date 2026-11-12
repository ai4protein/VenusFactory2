"""Fresh PBS imports keep discovery indexes out of the frozen source tree."""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agent.harness.pbs_goal_policy import freeze_settings
from agent.harness.pbs_goal_worker import _environment

ROOT = Path(__file__).resolve().parents[2]


def tree_hashes(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture(scope="module")
def source_copy(tmp_path_factory):
    source = tmp_path_factory.mktemp("frozen-library-source")
    for directory in ("src", "backend", "mcp_tools", "configs"):
        for path in (ROOT / directory).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".json", ".toml", ".yaml", ".yml"}:
                continue
            if "__pycache__" in path.parts:
                continue
            target = source / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    return source


@pytest.fixture
def record(tmp_path, source_copy):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    catalog = tmp_path / "free-models.json"
    catalog.write_text('{"models": []}')
    settings = {"runtime_root": str(tmp_path), "repository": str(source_copy), "environment": {},
                "execution_catalog": str(catalog)}
    value = {"source": {"root": str(source_copy)}, "settings": settings,
             "record_path": str(attempt / "submission.json"),
             "execution": {"python": sys.executable, "environment": {}, "extra_python_paths": []},
             "resources": {"ncpus": 2, "memory_gb": 16},
             "tool_settings": freeze_settings(settings, attempt)}
    path = attempt / "submission.json"
    path.write_text(json.dumps(value))
    return path, value


def fresh_python(source, program, *args, environment=None):
    # Isolated Python ignores inherited PYTHONPATH and never writes bytecode.
    dependencies = [entry for entry in sys.path if entry and "site-packages" in entry]
    bootstrap = "import sys\nsys.path[:0]=" + repr([str(source / "src"), str(source), *dependencies]) + "\n"
    result = subprocess.run([sys.executable, "-I", "-B", "-c", bootstrap + program, *map(str, args)],
                            env=environment, cwd=source, capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout)


def test_fresh_pbs_discovery_creates_only_private_library_and_keeps_public_scope(record, source_copy):
    path, value = record
    before = tree_hashes(source_copy)
    result = fresh_python(source_copy, r"""
import json, os
from pathlib import Path
from agent.harness.pbs_goal_worker import _environment
record_path = Path(sys.argv[1])
record = json.loads(record_path.read_text())
os.umask(0o077)
environment = _environment(record_path, record)
assert 'agent.harness.research_workspace' not in sys.modules
os.environ.clear()
os.environ.update(environment)
from agent.harness import research_workspace, capability_catalog
from mcp_tools.tools.discovery.generated_scope import GeneratedScope, search
library = research_workspace.LIBRARY
assert library == record_path.parent / 'runtime/research_capabilities'
assert search()['total'] == 0
assert GeneratedScope().references(lambda *args: 1) == []
# Synthetic metadata exercises unchanged curator/public filtering, not real model admission.
with capability_catalog.database(library) as connection:
    for number, visibility in enumerate(('public', 'private')):
        identity = str(number + 1) * 64
        card = {'id': identity, 'name': visibility + ' fixture', 'description': 'synthetic metadata',
                'author': 'fixture', 'version': '1', 'kind': 'tool', 'visibility': visibility,
                'review_status': 'approved', 'scope': {},
                'publication': {'policy': 'operator_selected.v1'} if visibility == 'public' else {}}
        connection.execute('INSERT INTO capabilities VALUES (?,?,?,?,?,?,?,?)',
                           (identity, 'fixture-goal', card['name'], '1', 'synthetic fixture',
                            json.dumps(card), '0' * 64, number))
page = search()
assert page['total'] == 1 and page['items'][0]['id'] == '1' * 64
references = GeneratedScope().references(lambda *args: 1)
assert references == [('1' * 64, 1)]
assert (library / 'catalog.sqlite3').is_file()
assert not (Path(record['source']['root']) / 'research_runs').exists()
print(json.dumps({'library': str(library), 'public_count': page['total']}))
""", path, environment={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"})
    assert result["library"] == str(path.parent / "runtime/research_capabilities")
    assert tree_hashes(source_copy) == before
    assert not (source_copy / "research_runs/rsi/research_capabilities").exists()


def test_pbs_library_cannot_be_replaced_by_parent_environment_or_config(record, monkeypatch, tmp_path):
    path, value = record
    foreign = tmp_path / "not-the-job-library"
    monkeypatch.setenv("VENUS_RESEARCH_CAPABILITY_ROOT", str(foreign))
    environment = _environment(path, value)
    assert environment["VENUS_RESEARCH_CAPABILITY_ROOT"] == str(path.parent / "runtime/research_capabilities")
    assert not foreign.exists()
    altered = copy.deepcopy(value)
    altered["execution"]["environment"]["VENUS_RESEARCH_CAPABILITY_ROOT"] = str(foreign)
    with pytest.raises(ValueError, match="Unsupported worker environment"):
        _environment(path, altered)


def test_unconfigured_web_cli_library_keeps_legacy_location_without_creating_it(source_copy):
    before = tree_hashes(source_copy)
    result = fresh_python(source_copy, """
import json
from agent.harness.research_workspace import LIBRARY
print(json.dumps({'library': str(LIBRARY), 'exists': LIBRARY.exists()}))
""", environment={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8"})
    assert result == {"library": str(source_copy / "research_runs/rsi/research_capabilities"), "exists": False}
    assert tree_hashes(source_copy) == before


def test_nonabsolute_host_library_override_fails_before_creating_files(source_copy):
    before = tree_hashes(source_copy)
    result = fresh_python(source_copy, """
import json
try:
    from agent.harness.research_workspace import LIBRARY
except ValueError as error:
    print(json.dumps({'rejected': 'absolute host-configured path' in str(error)}))
else:
    raise AssertionError('Relative library override accepted')
""", environment={"PATH": os.environ.get("PATH", ""), "LANG": "C.UTF-8",
                  "VENUS_RESEARCH_CAPABILITY_ROOT": "relative-library"})
    assert result == {"rejected": True}
    assert tree_hashes(source_copy) == before
