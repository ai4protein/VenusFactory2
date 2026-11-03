import pytest
from backend.web_v2.model_platform import Platform, write


def job(service, status='running'):
    value = dict(id='a'*32, owner='alice', kind='predict', status=status, progress=20,
                 created=1, payload={'name': 'ProSST-2048'}, steps=[])
    write(service.path('jobs', value['id']), value)
    service.register_job(value)
    return value


def test_polling_uses_index_metadata_and_immediately_reads_final_state(tmp_path, monkeypatch):
    service = Platform(tmp_path/'platform')
    value = job(service); value['created'] = 10**12
    write(service.path('jobs', value['id']), value)
    def unnecessary(*args, **kwargs):
        raise AssertionError('Polling must not reload models or rewrite registration')
    monkeypatch.setattr(service, 'register_job', unnecessary)
    monkeypatch.setattr(service, 'model', unnecessary)
    assert service.job_page('alice')['active'] == 1
    value.update(status='completed', progress=100, result={'preview': [], 'rows': 0})
    write(service.path('jobs', value['id']), value)
    result = service.job_page('alice')
    assert result['active'] == 0 and result['items'][0]['status'] == 'completed'
    assert result['items'][0]['title'] == 'ProSST-2048'
    assert service.job('alice', value['id'])['status'] == 'completed'
    assert service.job_page('bob')['items'] == []
    with pytest.raises(FileNotFoundError): service.job('bob', value['id'])


def test_completion_between_file_and_scheduler_reads_is_not_interrupted(tmp_path, monkeypatch):
    service = Platform(tmp_path/'platform'); value = job(service)
    value['queue_id'] = 'b'*32; write(service.path('jobs', value['id']), value)
    def complete(*args):
        value.update(status='completed', progress=100, result={'rows': 1})
        write(service.path('jobs', value['id']), value)
        return {'status': 'completed'}
    monkeypatch.setattr(service.queue, 'public', complete)
    result = service.job('alice', value['id'])
    assert result['status'] == 'completed' and result['result']['rows'] == 1
