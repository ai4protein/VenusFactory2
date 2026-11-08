import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from backend.web_v2 import research_predictions as predictions
from mcp_tools.scheduler import client
from mcp_tools.tools import capability_policy, runtime_tool_policy


@pytest.fixture
def online_policy(tmp_path, monkeypatch):
    store = capability_policy.CapabilityStore(tmp_path / 'capabilities.sqlite3')
    monkeypatch.setattr(runtime_tool_policy, 'is_online_mode', lambda: True)
    monkeypatch.setattr(capability_policy, 'CapabilityStore', lambda: store)
    monkeypatch.setenv('VENUS_QUEUE_OWNER', 'research-owner')
    monkeypatch.setenv('VENUS_QUEUE_JOB_ID', 'research-parent')
    return store


def spec(identity='a', model='ESM2'):
    return {'id': identity, 'task': 'Solubility', 'model': model, 'dataset': 'sol',
            'tool': 'predict_protein_function', 'adapter_path': '/checkpoints/sol/adapter'}


def disable_variant(store, value):
    store.set('model:' + value['id'], False, 0, {
        'tools': [value['tool']], 'adapter': 'function', 'task': value['task'],
        'model': value['model'], 'dataset': value['dataset'],
    }, 'test-admin')


def test_online_candidates_respect_variant_switches(tmp_path, monkeypatch, online_policy):
    root = tmp_path / 'repo'
    (root / 'src').mkdir(parents=True)
    config = {'web_ui': {
        'dataset_mapping_function': {'Solubility': ['sol']},
        'model_mapping_function': {'ESM2': 'esm2', 'Ankh': 'ankh'},
        'residue_mapping_function': {}, 'model_residue_mapping_function': {},
        'model_adapter_mapping_function': {'esm2': 'esm', 'ankh': 'ankh'},
    }}
    (root / 'src/constant.json').write_text(json.dumps(config))
    for adapter in ('esm', 'ankh'):
        folder = root / 'ckpt/sol' / adapter
        folder.mkdir(parents=True)
        (folder / 'lr5e-4_bt12k_ga8.pt').write_bytes(b'fixture')
        (folder / 'lr5e-4_bt12k_ga8.json').write_text('{}')
    disable_variant(online_policy, spec())
    candidates = predictions.candidates({}, root=root)
    assert [row['model'] for row in candidates] == ['Ankh']
    monkeypatch.setattr(predictions, 'candidates', lambda _: candidates)
    assert predictions.options()['available']
    monkeypatch.setattr(predictions, 'candidates', lambda _: [])
    assert not predictions.options()['available']


def test_online_panel_uses_owned_mcp_and_validates_actual_output(tmp_path, monkeypatch, online_policy):
    selected = spec()
    monkeypatch.setattr(predictions, 'candidates', lambda _: [selected])
    async def invoke(name, arguments, **kwargs):
        assert name == selected['tool']
        assert kwargs['owner'] == 'research-owner'
        assert kwargs['parent_job'] == 'research-parent'
        from pathlib import Path
        Path(arguments['output_file']).write_text('header,sequence,score\nquery,ACDEFGHIKLMN,0.75\n')
        return {'status': 'success'}
    call = AsyncMock(side_effect=invoke)
    monkeypatch.setattr(client, 'invoke', call)
    rows = asyncio.run(predictions.run_predictions(
        {'sequence': 'ACDEFGHIKLMN'}, lambda _: None, tmp_path / 'outputs'))
    assert call.await_count == 1
    assert rows[0]['status'] == 'completed'
    assert rows[0]['values'] == [{'score': '0.75'}]


def test_disabled_after_selection_never_reaches_mcp(tmp_path, monkeypatch, online_policy):
    selected = spec()
    monkeypatch.setattr(predictions, 'candidates', lambda _: [selected])
    disable_variant(online_policy, selected)
    call = AsyncMock()
    monkeypatch.setattr(client, 'invoke', call)
    rows = asyncio.run(predictions.run_predictions(
        {'sequence': 'ACDEFGHIKLMN'}, lambda _: None, tmp_path / 'outputs'))
    assert rows[0]['status'] == 'unavailable'
    call.assert_not_awaited()


def test_unowned_report_cannot_submit_predictions(tmp_path, monkeypatch, online_policy):
    monkeypatch.delenv('VENUS_QUEUE_OWNER')
    call = AsyncMock()
    monkeypatch.setattr(client, 'invoke', call)
    with pytest.raises(ValueError, match='owned research job'):
        asyncio.run(predictions.queued_prediction(spec(), tmp_path/'query.fasta', tmp_path/'result.csv', None))
    call.assert_not_awaited()


@pytest.mark.parametrize('detail,message', [
    ({'code': 'COMPUTE_DAILY_LIMIT_REACHED'}, 'Daily prediction allowance reached'),
    ({'code': 'COMPUTE_START_RATE_LIMIT'}, 'Prediction queue capacity reached'),
    ({}, 'Prediction queue capacity reached'),
])
def test_limits_stop_remaining_dispatches_and_keep_completed_results(tmp_path, monkeypatch, online_policy, detail, message):
    monkeypatch.setattr(predictions, 'candidates', lambda _: [spec(str(i)) for i in range(4)])
    calls = []
    async def invoke(name, arguments, **kwargs):
        calls.append(name)
        if len(calls) > 1:
            raise client.ToolJobError(429, 'limit', detail)
        from pathlib import Path
        Path(arguments['output_file']).write_text('score\n0.75\n')
        return {'status': 'success'}
    monkeypatch.setattr(client, 'invoke', invoke)
    rows = asyncio.run(predictions.run_predictions(
        {'sequence': 'ACDEFGHIKLMN'}, lambda _: None, tmp_path/'outputs'))
    assert len(calls) == 2
    assert [row['status'] for row in rows] == ['completed', 'deferred', 'deferred', 'deferred']
    assert rows[0]['values'] == [{'score': '0.75'}]
    assert all(row['message'] == message for row in rows[1:])


def test_mcp_rpc_preserves_structured_allowance_error(monkeypatch):
    detail = {'code': 'COMPUTE_DAILY_LIMIT_REACHED', 'resource': 'prediction'}
    response = SimpleNamespace(content=[SimpleNamespace(text=json.dumps({
        'error': {'code': 429, 'message': 'Limit reached', 'detail': detail}}))])
    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def call_tool(self, *args): return response
    monkeypatch.setattr(client, 'Client', lambda *args, **kwargs: FakeClient())
    monkeypatch.setattr(client, 'service_key', lambda: 'test-placeholder')
    with pytest.raises(client.ToolJobError) as result:
        asyncio.run(client.rpc('submit_tool', {}))
    assert result.value.code == 429
    assert result.value.detail == detail
    assert client.ToolJobError(404, 'Missing').detail == {}
