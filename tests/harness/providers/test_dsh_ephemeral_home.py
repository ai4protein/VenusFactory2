import sys
from pathlib import Path
from types import SimpleNamespace

from agent.harness.adapters import dsh_command


def test_sdk_boot_is_ephemeral_but_evidence_and_session_are_persisted(tmp_path):
    controls = tmp_path / 'control'
    workspace = tmp_path / 'session'
    ctx = SimpleNamespace(task=SimpleNamespace(workspace=workspace, capability=tmp_path / 'caps'),
                          runtimes={'python_root': sys.prefix, 'dsh_site': '/runtime/sdk'})
    command = dsh_command(ctx, controls)
    pairs = list(zip(command, command[1:]))
    assert ('--tmpfs', '/runtime_home') in pairs
    mounts = [(command[i+1], command[i+2]) for i, value in enumerate(command) if value == '--bind']
    assert mounts == [(str(workspace), '/workspace'), (str(controls), '/evidence')]
    assert '--die-with-parent' in command and '--clearenv' in command
    assert ('--setenv', 'HOME') in pairs
