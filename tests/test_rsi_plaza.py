import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
import pytest

from backend.web_v2.rsi_plaza import RsiPlaza


@pytest.fixture
def case(tmp_path, monkeypatch):
    monkeypatch.setattr('backend.web_v2.rsi_legacy_cases.cards', lambda query='': [])
    gid = 'a' * 32
    base = tmp_path / 'alice'
    folder = base / gid
    workspace = folder / 'workspace'
    workspace.mkdir(parents=True)
    (workspace / 'model.py').write_text('def run(x): return x\n')
    (workspace / 'README.md').write_text('# My model\n')
    (workspace / '.env').write_text('PRIVATE_SECRET=not-for-export')
    (folder / 'worker-control.json').write_text('private control')
    state = {'status': 'completed', 'definition': {'goal': {'objective': 'Study a model'}}, 'progress': {}, 'cycles': []}
    (folder / 'goal.json').write_text(json.dumps(state))
    meta = {'session_id': 'alice', 'spec': {'driver': 'autonomous_research.v1', 'objective': 'Study a model'}, 'created_at': 5, 'status': 'completed'}
    (base / (gid + '.json')).write_text(json.dumps(meta))
    def metadata(sid, identity):
        if sid != 'alice' or identity != gid: raise FileNotFoundError()
        return folder, base / (gid + '.json'), meta
    science = SimpleNamespace(session_folder=lambda sid: tmp_path / sid, metadata=metadata, live=lambda folder: False)
    service = RsiPlaza(science, tmp_path / 'library')
    return service, 'research:alice:' + gid, workspace


def test_private_case_preview_and_export(case):
    service, identity, workspace = case
    assert service.listing({'alice'})['total'] == 1
    detail = service.detail(identity, {'alice'})
    assert detail['readme'] == 'README.md' and detail['report'] == 'REPORT.md'
    assert service.preview(identity, {'alice'}, 'README.md')['text'] == '# My model\n'
    assert not any('.env' in f['path'] or 'worker-control' in f['path'] for f in detail['files'])
    archive = service.download(identity, {'alice'})
    with zipfile.ZipFile(archive) as z:
        assert 'model.py' in z.namelist() and 'REPORT.md' in z.namelist()
        assert not any('.env' in n or 'worker-control' in n for n in z.namelist())
    archive.close()


def test_cross_session_access_and_listing_are_denied(case):
    service, identity, _ = case
    assert service.listing({'bob'})['items'] == []
    for fn, args in ((service.detail, ()), (service.preview, ('model.py',)), (service.download, ())):
        with pytest.raises(FileNotFoundError): fn(identity, {'bob'}, *args)


@pytest.mark.parametrize('path', ['../goal.json', '/etc/passwd', '.env', 'x/../../goal.json', 'worker-control.json'])
def test_unsafe_or_nonrepository_paths_are_denied(case, path):
    service, identity, _ = case
    with pytest.raises((ValueError, FileNotFoundError)): service.preview(identity, {'alice'}, path)


def test_symlink_and_hardlink_cannot_export_host_files(case, tmp_path):
    import os
    service, identity, workspace = case
    secret = tmp_path / 'host-secret'
    secret.write_text('secret')
    (workspace / 'linked.py').symlink_to(secret)
    assert 'linked.py' not in [f['path'] for f in service.detail(identity, {'alice'})['files']]
    os.link(secret, workspace / 'hard.py')
    with pytest.raises(ValueError): service.detail(identity, {'alice'})


def test_listing_does_not_read_repository_files(case, monkeypatch):
    service, _, _ = case
    monkeypatch.setattr(service, 'resolve', lambda *args: pytest.fail('Listing must not read full projects'))
    monkeypatch.setattr('backend.web_v2.rsi_plaza.project_files', lambda *args, **kwargs: pytest.fail('Listing must not scan files'))
    assert service.listing({'alice'}, query='model', limit=1)['total'] == 1


def test_only_operator_curated_public_cases_are_listed(case):
    service, _, _ = case
    from agent.harness.capability_catalog import register
    from agent.harness.curated_publication import publication_record
    manifest = {'id': 'b'*64, 'name': 'Reviewed method', 'description': 'A small model', 'validation': {},
                'review_spec': {'kind': 'model', 'author': 'Researcher', 'version': '1',
                                'scope': {'proteins': ['plant'], 'properties': ['activity'], 'scenarios': ['lab']}}}
    receipt = {'receipt_sha256': 'c'*64}
    register(service.library, manifest, receipt, 'goal-1')
    assert service.listing(set())['items'] == []
    decision = publication_record(manifest, receipt, 'goal-1', author_email='author@example.org', curator='operator')
    register(service.library, manifest, receipt, 'goal-1', publication=decision)
    listed = service.listing(set())
    assert listed['total'] == 1 and listed['items'][0]['author_email'] == 'author@example.org'
    assert service.listing(set(), view='mine')['items'] == []


@pytest.mark.parametrize('online', [True, False])
def test_rsi_execution_is_retired_in_both_modes(monkeypatch, online):
    from mcp_tools.tools import runtime_tool_policy as policy
    from mcp_tools.models.evolved.policy import RSI_TOOL_NAMES
    monkeypatch.setattr(policy, 'is_online_mode', lambda: online)
    for name in RSI_TOOL_NAMES:
        assert not policy.is_tool_allowed(name)
        with pytest.raises(RuntimeError, match='RSI Plaza'):
            policy.assert_tool_allowed(name)
    from mcp_tools.tools.capability_policy import assert_model_enabled, CapabilityDisabled
    for identity in ('scientist:' + 'a'*64, 'capability:' + 'b'*64):
        with pytest.raises(CapabilityDisabled, match='RSI Plaza'): assert_model_enabled(identity)


def test_model_plaza_does_not_resolve_old_rsi_ids(tmp_path):
    from backend.web_v2.model_platform import Platform
    service = Platform(tmp_path)
    for identity in ('scientist:old', 'capability:old'):
        with pytest.raises(FileNotFoundError): service.model('alice', identity)
        with pytest.raises(ValueError): service.require_available({'id': identity, 'adapter': 'scientist'})


def test_legacy_operator_library_is_local_only(monkeypatch):
    from backend.web_v2 import rsi_legacy_cases as legacy
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_local_mode', lambda: False)
    assert legacy.cards() == []
    with pytest.raises(FileNotFoundError): legacy.documents('a'*64)


def test_unlisted_public_readme_is_never_read_or_exported(case, monkeypatch, tmp_path):
    service, _, _ = case
    from agent.harness.capability_catalog import register
    from agent.harness.curated_publication import publication_record
    bundle = tmp_path / 'public-bundle'
    bundle.mkdir()
    (bundle / 'model.py').write_text('def run(x): return x\n')
    (bundle / 'README.md').write_text('unreviewed-private-sentinel')
    (bundle / 'REPORT.md').write_text('unreviewed-report-sentinel')
    manifest = {'id':'d'*64, 'name':'Curated method', 'description':'Reviewed case', 'validation':{},
                'source':'model.py', 'files': {'model.py':'f'*64},
                'review_spec': {'kind':'model','author':'Researcher','version':'1',
                                'scope':{'proteins':['plant'],'properties':['activity'],'scenarios':['lab']}}}
    receipt = {'receipt_sha256':'e'*64}
    decision = publication_record(manifest, receipt, 'goal', author_email='author@example.org', curator='operator')
    register(service.library, manifest, receipt, 'goal', publication=decision)
    monkeypatch.setattr('agent.harness.research_workspace.admitted_manifest', lambda *args: (bundle, manifest, receipt))
    identity = 'capability:' + manifest['id']
    detail = service.detail(identity, set())
    assert {r['path'] for r in detail['files']} == {'README.md', 'REPORT.md', 'model.py'}
    archive = service.download(identity, set())
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {r['path'] for r in detail['files']}
        for path in ('README.md', 'REPORT.md', 'model.py'):
            text = service.preview(identity, set(), path)['text']
            assert 'unreviewed' not in text
            assert z.read(path).decode() == text
    archive.close()


def test_streamed_archive_budget_rechecks_growing_files(case, monkeypatch):
    service, identity, workspace = case
    (workspace / 'model.py').write_text('x'*10000)
    # Model a file that grew after metadata was inspected.
    monkeypatch.setattr(service, '_names', lambda *args: [{'path':'model.py','size':1}])
    monkeypatch.setattr('backend.web_v2.rsi_plaza.MAX_ARCHIVE', 5000)
    with pytest.raises(ValueError, match='archive budget'):
        service.download(identity, {'alice'})


def test_local_legacy_model_exports_recorded_artifacts_without_execution(tmp_path, monkeypatch):
    from backend.web_v2 import rsi_legacy_cases as legacy
    monkeypatch.setattr('mcp_tools.tools.runtime_tool_policy.is_local_mode', lambda: True)
    key = {'scope': {'protein':'plant protein','target':'activity'}, 'plugin_id':'fixture.v1'}
    source = 'def predict(rows): return [0 for row in rows]\n'
    publication = {'key':key, 'origin':{'stage':'activity'},
                   'packet':{'model':{'recipe':{'source_code':source},'parameters':{'bias':0}},
                             'assessment':{'metrics':{'error':1}}}}
    reads = []
    store = SimpleNamespace(root=tmp_path / 'legacy', inspect_publication=lambda identity: (reads.append(identity) or publication))
    monkeypatch.setattr(legacy, 'library', lambda: store)
    monkeypatch.setattr(legacy, '_metadata', lambda *args: [{'entry_id':'f'*64,'key':key,'origin':{'stage':'activity'}}])
    science = SimpleNamespace(session_folder=lambda sid: tmp_path / sid)
    service = RsiPlaza(science, tmp_path / 'curated')
    listing = service.listing(set())
    assert listing['total'] == 1 and not reads
    identity = listing['items'][0]['id']
    detail = service.detail(identity, set())
    assert detail['status'] == 'recorded' and detail['visibility'] == 'private'
    assert service.preview(identity, set(), 'model.py')['text'] == source
    archive = service.download(identity, set())
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {r['path'] for r in detail['files']}
        assert z.read('model.py').decode() == source
        assert json.loads(z.read('model.json'))['parameters']['bias'] == 0
    archive.close()
