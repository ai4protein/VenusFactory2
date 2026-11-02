import json
from unittest.mock import Mock
import pytest
from backend.web_v2.job_queue import JobQueue
from backend.web_v2.research_jobs import ResearchJobs, write

def fixture(tmp_path, status='interrupted', calls=4):
    q = JobQueue(tmp_path/'queue'); jobs = ResearchJobs(tmp_path/'reports', q); identity = 'a'*32
    qid = q.submit('alice', 'protein-research', 'research'); q.finish(qid, status, 'Worker interrupted')
    row = {'id': identity, 'owner': 'alice', 'created': 1, 'title': 'Protein', 'queue_id': qid,
           'payload': {'model_id': 'free', 'analyses': ['properties'], 'sequence': 'MKWV'}}
    write(jobs.folder(identity)/'request.json', row)
    write(jobs.folder(identity)/'state.json', {'model_calls': calls, 'report': 'Saved partial result'})
    jobs.records.register('report', identity, 'alice', 'research', 'Protein', 1)
    return jobs, identity

def test_resume_uses_same_identity_owner_and_checkpoint(tmp_path, monkeypatch):
    jobs, identity = fixture(tmp_path)
    monkeypatch.setattr(jobs, 'validate', lambda payload: None)
    launched = []
    def launch(command, **kwargs):
        launched.append(command)
        return jobs.queue.submit(kwargs['owner'], kwargs['kind'], kwargs['resource']), None
    monkeypatch.setattr('backend.web_v2.research_jobs.launch', launch)
    admit = Mock()
    result = jobs.resume('alice', identity, admit=admit)
    assert result['id'] == identity and result['status'] == 'queued'
    assert result['report'] == 'Saved partial result'
    assert result['model_calls'] == 4
    assert launched[0][-1] == identity
    assert jobs.read(identity)['resume_count'] == 1
    admit.assert_called_once_with(identity)
    jobs.resume('alice', identity, admit=admit)  # double-click never starts another worker
    assert len(launched) == 1
    with pytest.raises(FileNotFoundError): jobs.resume('bob', identity)

@pytest.mark.parametrize('status,calls', [('completed',4), ('cancelled',4), ('failed',18)])
def test_resume_rejects_finished_cancelled_or_exhausted_work(tmp_path, status, calls):
    jobs, identity = fixture(tmp_path, status, calls)
    with pytest.raises(ValueError): jobs.resume('alice', identity)

def test_resume_rechecks_free_model_policy_before_enqueue(tmp_path, monkeypatch):
    jobs, identity = fixture(tmp_path)
    def invalid(payload): raise ValueError('Choose a configured free model')
    monkeypatch.setattr(jobs, 'validate', invalid)
    launch = Mock(); monkeypatch.setattr('backend.web_v2.research_jobs.launch', launch)
    with pytest.raises(ValueError): jobs.resume('alice', identity)
    launch.assert_not_called()


def test_resume_preserves_activity_ids_and_does_not_repeat_saved_transitions():
    from backend.web_v2.research_activity import ActivityJournal
    old = [{'id': 501, 'at': 123, 'kind': 'phase', 'key': 'phase', 'status': 'chapters', 'label': '', 'job_number': ''}]
    journal = ActivityJournal(old)
    events = journal.observe({'phase': 'chapters', 'chapters': [{'id': 'identity', 'title': 'Identity', 'status': 'completed'}]})
    assert events[0] == old[0]
    assert [e['id'] for e in events] == [501, 502]
    assert len(journal.observe({'phase': 'chapters', 'chapters': [{'id': 'identity', 'status': 'completed'}]})) == 2
