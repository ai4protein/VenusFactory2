import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).resolve().parents[2] / 'scripts/rsi/evaluation/audit_shared_publications.py'
_spec = importlib.util.spec_from_file_location('publication_reaudit', _path)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


def test_checkpoint_rejects_changed_evidence_and_record(tmp_path):
    path = tmp_path / 'batch.json'
    binding = {'goal_tree_sha256': 'original'}
    record = {'binding': binding, 'status': 'passed', 'verified': []}
    audit.write(path, {'record': record, 'sha256': audit.digest(record)})
    assert audit.completed_checkpoint(path, binding) == record
    with pytest.raises(ValueError, match='evidence changed'):
        audit.completed_checkpoint(path, {'goal_tree_sha256': 'altered'})
    packet = audit.read(path)
    packet['record']['verified'] = [{'scope_id': 'injected'}]
    audit.write(path, packet)
    with pytest.raises(ValueError, match='checkpoint changed'):
        audit.completed_checkpoint(path, binding)


def test_failed_attempt_is_retryable_but_never_reused_as_success(tmp_path):
    path = tmp_path / 'batch.json'
    record = {'binding': {'source': 'fixed'}, 'status': 'failed', 'verified': []}
    audit.write(path, {'record': record, 'sha256': audit.digest(record)})
    assert audit.completed_checkpoint(path, record['binding']) is None


def test_mutating_data_inside_evidence_changes_fingerprint(tmp_path):
    f = tmp_path / 'data' / 'training.json'; f.parent.mkdir()
    f.write_text('[1,2]')
    first = audit.tree_hash(tmp_path)
    f.write_text('[1,3]')
    assert audit.tree_hash(tmp_path) != first
    (tmp_path / 'link').symlink_to(f)
    with pytest.raises(ValueError, match='Symlinked'):
        audit.tree_hash(tmp_path)


def test_partial_or_failed_batches_do_not_establish_complete_verification():
    good = {'status': 'passed', 'verified': [{'scope_id': 'a', 'program_sha256': 'p'}]}
    bad = {'status': 'failed', 'verified': [{'scope_id': 'b', 'program_sha256': 'q'}]}
    partial = audit.summary([good], 2)
    assert not partial['finished'] and not partial['all_publication_checks_passed']
    report = audit.summary([good, bad], 2)
    assert report['finished'] and not report['all_publication_checks_passed']
    assert report['verified_scope_count'] == 1 and report['distinct_verified_programs'] == 1
    assert report['new_autonomous_models'] == 0 and not report['final_labels_scored']


def test_shared_programs_do_not_multiply_algorithm_count_or_duplicate_scopes():
    good = {'status': 'passed', 'verified': [{'scope_id': 'a', 'program_sha256': 'p'}, {'scope_id': 'b', 'program_sha256': 'p'}]}
    report = audit.summary([good], 1)
    assert report['verified_scope_count'] == 2 and report['distinct_verified_programs'] == 1
    with pytest.raises(ValueError, match='Repeated scientific scope'):
        audit.summary([good, good], 2)


@pytest.mark.parametrize('layout', ['flat', 'packaged'])
def test_frozen_runtime_layouts_are_imported_without_rewriting(tmp_path, monkeypatch, layout):
    from types import SimpleNamespace
    base=tmp_path/'src/agent/rsi'
    source=base/('campaigns' if layout=='packaged' else '')/'shared_model_campaign.py'
    source.parent.mkdir(parents=True);source.write_text('# frozen source')
    original=source.read_bytes()
    module=SimpleNamespace(__file__=str(source))
    imported=[]
    def load(name):
        imported.append(name)
        return module
    monkeypatch.setattr(audit.importlib,'import_module',load)
    assert audit.runtime_module(tmp_path,'campaigns','shared_model_campaign') is module
    assert imported == ['agent.rsi.'+('campaigns.' if layout=='packaged' else '')+'shared_model_campaign']
    assert source.read_bytes()==original


def test_ambient_runtime_cannot_impersonate_frozen_source(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source=tmp_path/'frozen/src/agent/rsi/shared_model_campaign.py'
    source.parent.mkdir(parents=True);source.touch()
    foreign=tmp_path/'current/src/agent/rsi/shared_model_campaign.py'
    monkeypatch.setattr(audit.importlib,'import_module',lambda _:SimpleNamespace(__file__=str(foreign)))
    with pytest.raises(ValueError,match='different runtime'):
        audit.runtime_module(tmp_path/'frozen','campaigns','shared_model_campaign')
