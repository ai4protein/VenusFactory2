import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('repository_check', Path(__file__).resolve().parents[1] / 'scripts/check_repository.py')
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


@pytest.mark.parametrize('name', [
    f'uploads/session.{suffix}{sidecar}'
    for suffix in ['db', 'sqlite', 'sqlite3']
    for sidecar in ['', '-wal', '-shm', '-journal']
])
def test_sqlite_state_is_rejected_even_if_force_added(name):
    assert checks.inspect_bytes(name, b'conversation state')


def test_schema_and_documentation_are_allowed():
    assert not checks.inspect_bytes('backend/schema.sql', b'CREATE TABLE jobs (id TEXT)')
    assert not checks.inspect_bytes('docs/sqlite.md', b'Queue storage')
