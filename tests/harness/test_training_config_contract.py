import json
from pathlib import Path

import pytest


@pytest.fixture
def config_env(tmp_path, monkeypatch):
    from mcp_tools.models.factory.training import train_operations as ops
    monkeypatch.setenv('VENUS_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    monkeypatch.setenv('WEBUI_V2_MODE', 'local')
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_online_mode', lambda: False)
    def forbidden(*args, **kwargs):
        pytest.fail('Training configuration must not invoke a second LLM or make an HTTP request')
    monkeypatch.setattr('requests.sessions.Session.request', forbidden)
    configs = tmp_path / 'configs'
    configs.mkdir()
    monkeypatch.setattr(ops, 'get_save_path', lambda *args: str(configs))
    csv = tmp_path / 'train.csv'
    csv.write_text('sequence,target\nACDEFGHIKLMNPQ,0\nACDEFGHIKLMNPR,1\nACDEFGHIKLMNPS,0\nACDEFGHIKLMNPT,1\n')
    return ops, csv, configs


def test_training_configuration_is_explicit_and_offline(config_env):
    from mcp_tools.models.factory.training.tools_agent import generate_training_config_tool
    _, csv, _ = config_env
    result = json.loads(generate_training_config_tool.invoke({
        'csv_file': str(csv), 'parameters': {'num_epochs': 2, 'batch_size': 4, 'learning_rate': 0.002}}))
    assert result['success'], result
    config = json.loads(Path(result['config_path']).read_text())
    assert config['plm_model'] == 'ESM2-8M'
    assert config['num_epochs'] == 2 and config['batch_size'] == 4 and config['learning_rate'] == 0.002
    assert config['sequence_column_name'] == 'sequence' and config['label_column_name'] == 'target'
    assert result['dataset_info']['num_labels'] == config['num_labels'] == 2
    assert result['configuration_source'] == 'explicit_parameters_and_defaults'
    for key in ['plm_model', 'num_epochs', 'learning_rate', 'batch_size']:
        assert result['resolved_parameters'][key] == config[key]


@pytest.mark.parametrize('parameters', [
    {'output_dir': '/unexpected'}, {'num_epochs': 0}, {'batch_size': True},
    {'learning_rate': float('nan')}, {'max_grad_norm': float('inf')},
    {'num_workers': -1}, [],
])
def test_bad_parameters_do_not_write_a_configuration(config_env, parameters):
    ops, csv, configs = config_env
    result = json.loads(ops.process_csv_and_generate_config(str(csv), user_overrides=parameters))
    assert not result['success'] and result['error']
    assert not list(configs.iterdir())


def test_legacy_free_text_cannot_launch_hidden_model_request(config_env):
    ops, _, configs = config_env
    result = json.loads(ops.process_csv_and_generate_config(dataset_path='not/downloaded', user_requirements='train for 5 epochs'))
    assert not result['success'] and 'explicit parameters' in result['error']
    assert not list(configs.iterdir())


@pytest.mark.asyncio
async def test_fastmcp_wrapper_accepts_same_explicit_parameters(config_env):
    from fastmcp import Client
    from mcp_tools.models.factory.training.tools_mcp import mcp
    _, csv, _ = config_env
    async with Client(mcp) as client:
        tools = await client.list_tools()
        schema = next(t.inputSchema for t in tools if t.name == 'generate_training_config')
        assert 'parameters' in schema['properties']
        response = await client.call_tool('generate_training_config', {
            'csv_file': str(csv), 'parameters': {'num_epochs': 3, 'batch_size': 2}})
        result = json.loads(response.content[0].text)
    assert result['success'], result
    assert result['resolved_parameters']['num_epochs'] == 3
    assert json.loads(Path(result['config_path']).read_text())['batch_size'] == 2
