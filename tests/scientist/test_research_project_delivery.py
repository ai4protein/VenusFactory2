import asyncio
import hashlib
import json
import os
import zipfile
from pathlib import Path
import pytest

from backend.web_v2.research_delivery import export_project, report
from backend.web_v2.science_goals import ScienceGoalService, ROOT


def state():
    return {'definition': {'goal': {'objective': 'Build a protein model'}}, 'status': 'waiting_for_input',
            'progress': {'plan': {'steps': [{'title': 'Collect measurements', 'acceptance': 'Measured labels', 'status': 'pending'}]},
                         'questions': ['Please supply measured solubility data.']},
            'cycles': [{'index': 0, 'status': 'completed', 'summary': 'No measured labels found.'}]}


def test_project_exports_tree_evidence_and_checksums_without_control_files(tmp_path):
    work = tmp_path / 'workspace'; (work / 'src').mkdir(parents=True)
    (work / 'README.md').write_text('Run python src/predict.py')
    (work / 'src/predict.py').write_text('print(1)')
    (work / '.env').write_text('SECRET=hidden')
    (tmp_path / 'profile.json').write_text('private execution profile')
    (work / 'outside').symlink_to(tmp_path / 'profile.json')
    with export_project(tmp_path, state()) as file, zipfile.ZipFile(file) as z:
        assert 'project/src/predict.py' in z.namelist()
        assert all('profile' not in n and '.env' not in n and 'outside' not in n for n in z.namelist())
        assert 'waiting_for_input' in z.read('REPORT.md').decode()
        manifest = json.loads(z.read('checksums.json'))
        assert set(manifest) == set(z.namelist()) - {'checksums.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest() == v for n,v in manifest.items())
        assert json.loads(z.read('research/status.json'))['research_completed'] is False


def test_hardlink_cannot_export_other_session_file(tmp_path):
    (tmp_path / 'workspace').mkdir()
    secret = tmp_path / 'another-session.txt'; secret.write_text('not mine')
    os.link(secret, tmp_path / 'workspace/leak.txt')
    with pytest.raises((ValueError, OSError)):
        export_project(tmp_path, state())


@pytest.mark.parametrize('mode,port', [('local','8084'),('online','8085')])
def test_scientist_worker_keeps_its_own_deployment_configuration(tmp_path, monkeypatch, mode, port):
    from backend.web_v2.runtime_paths import AREAS
    monkeypatch.setenv('WEBUI_V2_MODE', mode)
    monkeypatch.setenv('MCP_HTTP_PORT', port)
    for key,relative in AREAS.items(): monkeypatch.setenv(key, str(tmp_path / mode / relative))
    monkeypatch.setenv('UNRELATED_ACCOUNT_SECRET', 'must-not-leak')
    svc = ScienceGoalService(ROOT, state_root=tmp_path/'goals')
    env = svc.environment()
    assert env['MCP_HTTP_PORT'] == port and env['WEBUI_V2_MODE'] == mode
    assert all(env[k] == os.environ[k] for k in AREAS)
    assert 'UNRELATED_ACCOUNT_SECRET' not in env


def test_nested_adapter_reuses_slot_without_gpu_deadlock(tmp_path, monkeypatch):
    from backend.web_v2.job_queue import JobQueue, resource_slot, run_in_slot
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT', str(tmp_path/'queue'))
    monkeypatch.delenv('VENUS_QUEUE_JOB_ID', raising=False)
    monkeypatch.setenv('VENUS_QUEUE_GPU_SLOTS', '1')
    async def inner():
        async with resource_slot('gpu') as identity:
            return identity
    identity = run_in_slot('gpu', lambda: asyncio.run(asyncio.wait_for(inner(), 3)))
    q = JobQueue()
    assert q.get(identity)['status'] == 'completed'
    with q.connection() as c: assert c.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1
