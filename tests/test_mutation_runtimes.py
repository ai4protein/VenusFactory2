import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import mcp_tools.models.pretrained.runners.mutation as runtime


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / 'input.fasta'
    source.write_text('>test\nMKT\n')
    return source, tmp_path / 'output.csv'


def test_configured_interpreter_paths_and_secret_boundary(tmp_path, inputs, monkeypatch):
    monkeypatch.setattr(runtime, 'PROJECT', tmp_path)
    monkeypatch.setenv('DMXAPI_API_KEY', 'must-not-pass')
    monkeypatch.setenv('PYTHONPATH', '/unrelated/environment')
    monkeypatch.setenv('CUSTOM_PRIVATE_VALUE', 'must-not-pass')
    python = tmp_path / 'env/bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    weights = tmp_path / 'weights'
    weights.mkdir()
    command = runtime.build_command('ESM-1v', 'sequence', *inputs, settings={
        'esm1v': {'python': str(python), 'weights': str(weights), 'enabled': True,
                  'cache': str(tmp_path / 'cache'), 'timeout_seconds': 23}})
    assert command.argv[0] == str(python)  # preserve the venv symlink
    assert command.argv[-1] == str(weights)
    assert command.timeout == 23
    assert not {'DMXAPI_API_KEY', 'CUSTOM_PRIVATE_VALUE'} & command.env.keys()
    assert '/unrelated/environment' not in command.env['PYTHONPATH']
    assert command.env['HF_HOME'] == str(tmp_path / 'cache/huggingface')
    assert command.env['PYTHON_DOTENV_DISABLED'] == '1'


@pytest.mark.parametrize('profile', [{'enabled': False}, {'enabled': True, 'python': '/not/installed/python'},
                                    {'enabled': True, 'weights': '/not/installed/weights'}])
def test_explicit_unavailable_profile_never_falls_back(profile, inputs):
    with pytest.raises(ValueError):
        runtime.build_command('ESM-1v', 'sequence', *inputs, settings={'esm1v': profile})


def test_input_types_and_known_adapter_allowlist(inputs):
    with pytest.raises(ValueError):
        runtime.build_command('ESM-IF1', 'sequence', *inputs, settings={})
    with pytest.raises(ValueError):
        runtime.build_command('../../script', 'sequence', *inputs, settings={})
    command = runtime.build_command('ESM-IF', 'structure', *inputs, settings={})
    assert command.profile == 'esmif1'
    assert '--pdb_file' in command.argv
    assert command.argv[0] == sys.executable


def test_runtime_executes_in_configured_workdir_without_secrets(inputs, tmp_path):
    code = "import json,os;open('result.json','w').write(json.dumps({'cwd':os.getcwd(),'secret':os.getenv('DMXAPI_API_KEY')}))"
    work = tmp_path / 'job'
    command = runtime.RuntimeCommand([sys.executable, '-c', code], {}, work, 10, 'esm1v')
    runtime.execute(command)
    assert json.loads((work / 'result.json').read_text()) == {'cwd': str(work), 'secret': None}


def test_timeout_terminates_runtime(inputs, tmp_path):
    command = runtime.RuntimeCommand([sys.executable, '-c', 'import time;time.sleep(30)'], {}, tmp_path, 1, 'esm1v')
    with pytest.raises(subprocess.TimeoutExpired):
        runtime.execute(command)


def test_ensemble_restores_input_order_and_duplicates(inputs, monkeypatch):
    def fake_execute(command):
        output = Path(command.argv[command.argv.index('--output_csv') + 1])
        output.write_text('mutant,esm1v_score\nK2A,2\nM1A,1\n')
    monkeypatch.setattr(runtime, 'execute', fake_execute)
    monkeypatch.setattr(runtime, 'load_runtime_settings', lambda: {})
    assert runtime.score_mutations('ESM-1v', inputs[0], ['M1A', 'K2A', 'M1A']) == [1, 2, 1]


def test_public_runner_dispatches_selected_profile(inputs, monkeypatch, tmp_path):
    from mcp_tools.common import prediction_runners
    calls = []
    def fake_execute(command):
        calls.append(command.profile)
        Path(command.argv[command.argv.index('--output_csv') + 1]).write_text('mutant,esm1v_score\nM1A,1\n')
    monkeypatch.setattr(runtime, 'execute', fake_execute)
    monkeypatch.setattr(runtime, 'load_runtime_settings', lambda: {})
    monkeypatch.setenv('TEMP_OUTPUTS_DIR', str(tmp_path))
    message, result = prediction_runners.run_zero_shot_prediction('sequence', 'ESM-1v', str(inputs[0]))
    assert calls == ['esm1v']
    assert result.to_dict('records') == [{'mutant': 'M1A', 'esm1v_score': 1}]


def test_backend_and_mcp_import_without_archived_ui():
    root = Path(__file__).resolve().parents[1]
    code = '''
import sys,importlib.abc
class NoLegacyUI(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in {'web', 'gradio', 'archive'} or name.startswith('src.web.'):
            raise RuntimeError('Retired UI import: '+name)
sys.meta_path.insert(0, NoLegacyUI())
import backend.api_server
from mcp_tools.models.pretrained.mutation.models import mutation_operations
from mcp_tools.tools.predict.workflows import handle_protein_residue_function_prediction
result=list(handle_protein_residue_function_prediction('',None,False,'',None))
assert len(result)==1 and result[0][3]=={'visible':False}
from mcp_tools.common.common_utils import get_project_root
assert get_project_root().joinpath('frontend').is_dir()
'''
    done = subprocess.run([sys.executable, '-c', code], cwd=root, capture_output=True, text=True,
                          env={**os.environ, 'PYTHONPATH': str(root) + os.pathsep + str(root / 'src')}, timeout=90)
    assert done.returncode == 0, done.stderr[-6000:]
