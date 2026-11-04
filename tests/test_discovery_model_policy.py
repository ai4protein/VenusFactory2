import json
from pathlib import Path
import pytest

from backend.web_v2 import remote_discovery, workflow_tools
from backend.web_v2.model_platform import Platform
import mcp_tools.models.pretrained.runners.mutation as mutation
from mcp_tools.tools import runtime_tool_policy as policy
from mcp_tools.tools.research import services, alignment


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('VENUS_RUNTIME_ROOT', str(tmp_path / 'runtime'))
    import mcp_tools.models.checkpoint_cache as checkpoint_cache
    monkeypatch.setattr(checkpoint_cache, 'cached_model', lambda model_id, cache=None: model_id)


@pytest.mark.parametrize('size', mutation.PROSST_VOCABULARIES)
def test_variant_routes_correct_checkpoint_and_tokenizer(size, tmp_path):
    source = tmp_path / 'input.pdb'; source.write_text('END\n')
    common = {'weights': str(tmp_path / 'default-2048'), 'model_id': 'wrong/default'}
    (tmp_path / 'default-2048').mkdir()
    command = mutation.build_command(f'ProSST-{size}', 'structure', source, tmp_path / 'out.csv', settings={'prosst': common})
    assert command.argv[command.argv.index('--structure_vocab_size') + 1] == str(size)
    actual = command.argv[command.argv.index('--model_name') + 1]
    assert actual == (str(tmp_path / 'default-2048') if size == 2048 else f'AI4Protein/ProSST-{size}')


def test_variant_profile_and_missing_tokenizer(tmp_path):
    source = tmp_path / 'input.pdb'; source.write_text('END\n')
    settings = {'prosst': {'variants': {'128': {'enabled': False}}}}
    with pytest.raises(ValueError, match='disabled'):
        mutation.build_command('ProSST-128', 'structure', source, tmp_path / 'o', settings=settings)
    settings = {'prosst': {'structure_weights': str(tmp_path)}}
    (tmp_path / 'AE.pt').touch(); (tmp_path / '2048.joblib').touch()
    with pytest.raises(ValueError, match='tokenizer'):
        mutation.build_command('ProSST-128', 'structure', source, tmp_path / 'o', settings=settings)


def test_online_policy_keeps_scoring_remote_search_but_not_local_discovery(monkeypatch, tmp_path):
    monkeypatch.setattr(policy, 'is_online_mode', lambda: True)
    cards = Platform(tmp_path / 'platform').official()
    wanted = [c for c in cards if c['adapter'] == 'structure-evolution' and (c['name'].startswith('ProSST-') or c['name'].startswith('VenusREM'))]
    assert len(wanted) == 7 and all(c['available'] for c in wanted)
    assert not next(c for c in cards if c['name'] == 'VenusMine')['available']
    assert not next(c for c in cards if c['name'] == 'ProSST-2048 · Backbone')['available']
    for name in ('bindcraft', 'boltzgen'):
        card = next(c for c in cards if c['adapter'] == 'external:' + name)
        assert not card['available'] and card['disabled_reason'] == 'local_only'
        assert not policy.is_tool_allowed(name + ('_dock' if name == 'diffdock' else '_design'))
    diffdock = next(c for c in cards if c['adapter'] == 'external:diffdock')
    assert diffdock.get('disabled_reason') != 'local_only' and policy.is_tool_allowed('diffdock_dock')
    for name in ('query_ncbi_blast','query_foldseek_remote'):
        card = workflow_tools.card(name)
        assert card['category'] == 'discovery' and card['available']
        assert not {'job_id', 'sequence', 'structure_pdb', 'accession'} & card['contract']['properties'].keys()


def test_admin_disabling_remote_tool_also_blocks_venusrem(monkeypatch, tmp_path):
    from mcp_tools.tools.capability_policy import CapabilityStore
    monkeypatch.setattr(policy, 'is_online_mode', lambda: True)
    CapabilityStore().set('tool:query_foldseek_remote', False, 0, {}, 'test')
    card = next(c for c in Platform(tmp_path / 'p').official() if c['name'].startswith('VenusREM'))
    assert not card['available'] and card['disabled_reason'] == 'admin_disabled'
    with pytest.raises(Exception, match='disabled|unavailable|Unavailable'):
        remote_discovery.run('query_foldseek_remote', {'kind': 'structure', 'structure': ''}, {}, tmp_path)


def test_disabled_prosst_variant_cannot_be_bypassed_by_alias(tmp_path):
    from mcp_tools.tools.capability_policy import CapabilityStore, selector, CapabilityDisabled
    card = next(c for c in Platform(tmp_path / 'p').official() if c['name'] == 'ProSST-2048')
    store = CapabilityStore()
    store.set('model:' + card['id'], False, 0, selector(card), 'test')
    with pytest.raises(CapabilityDisabled):
        store.assert_call('zero_shot_mutation_structure_prediction', {'model_name': 'ProSST'})
    store.assert_call('zero_shot_mutation_structure_prediction', {'model_name': 'ProSST-128'})


@pytest.mark.parametrize('state', ['completed', 'empty', 'pending', 'unavailable'])
def test_remote_workflow_does_not_turn_pending_into_success(state, monkeypatch, tmp_path):
    calls = []; progress = []
    async def service(name, context, update, cache_dir):
        calls.append((name, context, cache_dir)); update({'status': 'waiting', 'job_id': 'remote123'})
        return {'status': state, 'job_id': 'remote123', 'url': 'https://blast.ncbi.nlm.nih.gov/',
                'data': {'hits': [{'accession': 'P12345', 'evalue': 1e-8}] if state == 'completed' else []}}
    monkeypatch.setattr(services, 'run_service', service)
    data = {'kind': 'table', 'rows': [{'id': 'test', 'sequence': 'ACDEFGHIKLMNPQ'}]}
    run = lambda: remote_discovery.run('query_ncbi_blast', data, {}, tmp_path, lambda *p: progress.append(p))
    if state in {'completed', 'empty'}:
        result = run(); assert result['rows'][0]['search_status'] == state
        assert result['rows'][0]['source_url'].startswith('https://')
        assert result['rows'][0]['hit_count'] == (1 if state == 'completed' else 0)
    else:
        with pytest.raises(ValueError, match=state): run()
    assert len(calls) == 1 and calls[0][0] == 'blast' and progress


def test_invalid_batch_validated_before_external_submission(monkeypatch, tmp_path):
    async def unexpected(*a, **kw): pytest.fail('No external request for invalid batch')
    monkeypatch.setattr(services, 'run_service', unexpected)
    data = {'kind': 'table', 'rows': [{'sequence': 'ACDEFGHIKLMN'}, {'sequence': 'INVALID*'}]}
    with pytest.raises((ValueError, services.ServiceError)):
        remote_discovery.run('query_ncbi_blast', data, {}, tmp_path)
    with pytest.raises(ValueError, match='five'):
        remote_discovery.run('query_ncbi_blast', {**data, 'rows': data['rows'] * 3}, {}, tmp_path)


def test_venusrem_alignment_has_query_coordinates_and_rejects_invalid_hits():
    hits = [{'qAln': 'AC-DE', 'dbAln': 'ACFDE', 'qStartPos': 2, 'qEndPos': 5, 'target': 'valid'},
            {'qAln': 'AC-DE', 'dbAln': 'AC', 'qStartPos': 2, 'qEndPos': 5, 'target': 'bad'}]
    assert alignment.alignment_rows(hits, 6) == {'-ACDE-': 'valid'}
    assert 'qAln' not in services.foldseek_hits({'results': [{'alignments': [hits]}]})[0]
    assert services.foldseek_hits({'results': [{'alignments': [hits]}]}, True)[0]['qAln'] == 'AC-DE'


def test_venusrem_uses_bounded_managed_runtime(tmp_path):
    source = tmp_path / 'input.pdb'; source.write_text('END\n')
    command = mutation.build_command('VenusREM (foldseek-based)', 'structure', source, tmp_path / 'out.csv', settings={})
    assert command.argv[1].endswith('/venusrem.py') and command.timeout <= 3600
    assert '--structure_vocab_size' not in command.argv and '--model_name' not in command.argv
