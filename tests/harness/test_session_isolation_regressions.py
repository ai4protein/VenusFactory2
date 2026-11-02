"""Exercise OS isolation and real file races with synthetic data, without an LLM."""
import asyncio
import io
import json
import os
from pathlib import Path
import zipfile

import pytest

from agent.harness.session_sandbox import execute, check_workspace
from agent.harness.research_workspace import ResearchWorkspace
from utils.scoped_files import open_file, read_bytes, write_bytes, atomic_writer
from backend.web_v2.research_files import archive_files, list_files


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['local','online'])
async def test_actual_guest_cannot_see_parent_credentials_or_control_fds(tmp_path,monkeypatch,mode):
    monkeypatch.setenv('WEBUI_V2_MODE',mode)
    monkeypatch.setenv('VENUS_SESSION_SANDBOX','bwrap')
    monkeypatch.setenv('PRIVATE_TEST_KEY','synthetic_parent_canary_19df0')
    work=tmp_path/'workspace';work.mkdir()
    (work/'probe.py').write_text('''import os,json
from pathlib import Path
for entry in Path('/proc').glob('[0-9]*/environ'):
    try: value=entry.read_bytes()
    except OSError: continue
    assert b'synthetic_parent_canary_19df0' not in value
assert not Path('/opt/venus-runtime/conda-meta').exists()
assert not Path('/opt/venus-runtime/.env').exists()
assert not Path('/etc/passwd').exists()
assert not Path('/home/tanyang').exists()
assert os.readlink('/proc/self/fd/1').startswith('pipe:')
assert os.readlink('/proc/self/fd/2').startswith('pipe:')
assert 'NoNewPrivs:\\t1' in Path('/proc/self/status').read_text()
assert b'--ro-bind' not in Path('/proc/1/cmdline').read_bytes()
print('isolated')
''')
    result=await execute(work,tmp_path/'control','python','probe.py',15)
    assert result['exit_code']==0,result
    assert result['stdout'].strip()=='isolated'


@pytest.mark.asyncio
async def test_real_scientific_analysis_and_persistent_artifacts(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_SESSION_SANDBOX','bwrap')
    work=tmp_path/'workspace';work.mkdir()
    (work/'data.csv').write_text('concentration,signal\n1,3\n2,5\n3,7\n4,9\n')
    (work/'analysis.py').write_text('''import json
import numpy as np,pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
data=pd.read_csv('data.csv');model=LinearRegression().fit(data[['concentration']],data.signal)
data['prediction']=model.predict(data[['concentration']]);data.to_csv('predictions.csv',index=False)
plt.plot(data.concentration,data.signal,'o-');plt.savefig('analysis.png')
json.dump({'slope':float(model.coef_[0]),'correlation':float(pearsonr(data.signal,data.prediction).statistic),'mass':ProteinAnalysis('ACDEFGHIK').molecular_weight()},open('summary.json','w'))
print('analysis complete')
''')
    result=await execute(work,tmp_path/'control','python','analysis.py',45)
    assert result['exit_code']==0,result
    assert json.loads((work/'summary.json').read_text())['slope']==pytest.approx(2)
    assert (work/'analysis.png').stat().st_size>100
    assert 'prediction' in (work/'predictions.csv').read_text()


@pytest.mark.asyncio
async def test_output_flood_is_bounded_and_terminated(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_SESSION_SANDBOX','bwrap')
    work=tmp_path/'w';work.mkdir()
    (work/'flood.py').write_text('import os\nwhile True: os.write(1,b"x"*65536)\n')
    with pytest.raises(ValueError,match='output limit'):
        await asyncio.wait_for(execute(work,tmp_path/'control','python','flood.py',10),15)
    assert sum(p.stat().st_size for p in (tmp_path/'control').iterdir() if p.suffix in {'.out','.err'})<=2*1024**2


@pytest.mark.asyncio
async def test_cancel_destroys_background_children(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_SESSION_SANDBOX','bwrap')
    work=tmp_path/'w';work.mkdir()
    (work/'run.py').write_text("import time\nopen('ready','w').close()\ntime.sleep(20)\nopen('late','w').close()")
    pending=asyncio.create_task(execute(work,tmp_path/'control','python','run.py',25))
    for _ in range(100):
        if (work/'ready').exists(): break
        await asyncio.sleep(.05)
    assert (work/'ready').exists()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError): await pending
    await asyncio.sleep(2.2)
    assert not (work/'late').exists()


@pytest.mark.asyncio
async def test_background_children_are_rejected_before_they_can_persist(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_SESSION_SANDBOX','bwrap')
    work=tmp_path/'w';work.mkdir()
    result=await execute(work,tmp_path/'control','shell','(sleep 2; echo bad > late) & echo done',10)
    assert result['exit_code']!=0 and ('fork' in result['stderr'].lower() or 'resource' in result['stderr'].lower())
    await asyncio.sleep(2.2)
    assert not (work/'late').exists()


@pytest.mark.parametrize('link',['symlink','hardlink','fifo'])
def test_links_and_special_files_never_enter_reads_or_archives(tmp_path,link):
    work=tmp_path/'w';work.mkdir();private=tmp_path/'private';private.write_text('private fixture')
    path=work/'data.txt'
    if link=='symlink':path.symlink_to(private)
    elif link=='hardlink':os.link(private,path)
    else:os.mkfifo(path)
    with pytest.raises((OSError,ValueError)):read_bytes(work,'data.txt')
    with pytest.raises((OSError,ValueError)):archive_files({'agent':work},['agent/data.txt'])
    with pytest.raises((OSError,ValueError)):check_workspace(work)
    assert list_files({'agent':work})['files']==[]


def test_open_descriptor_remains_owned_after_leaf_and_parent_swap(tmp_path):
    work=tmp_path/'w';work.mkdir();sub=work/'input';sub.mkdir();(sub/'file.txt').write_text('owned')
    private=tmp_path/'private';private.mkdir();(private/'file.txt').write_text('secret')
    with open_file(work,'input/file.txt') as stream:
        sub.rename(work/'old');sub.symlink_to(private,target_is_directory=True)
        assert stream.read()==b'owned'
    with pytest.raises(OSError):read_bytes(work,'input/file.txt')


def test_atomic_writer_cannot_follow_replaced_parent(tmp_path):
    work=tmp_path/'w';work.mkdir();sub=work/'output';sub.mkdir()
    private=tmp_path/'private';private.mkdir();(private/'result.txt').write_text('private')
    with atomic_writer(work,'output/result.txt') as stream:
        sub.rename(work/'old');sub.symlink_to(private,target_is_directory=True)
        stream.write(b'owned')
    assert (private/'result.txt').read_text()=='private'
    assert (work/'old/result.txt').read_text()=='owned'


@pytest.mark.asyncio
async def test_research_tools_do_not_read_or_overwrite_link_targets(tmp_path):
    work=tmp_path/'w';work.mkdir();private=tmp_path/'private';private.write_text('private')
    os.link(private,work/'data.txt')
    service=ResearchWorkspace(work,tmp_path/'control',{'steps':[]})
    with pytest.raises(ValueError):await service.read_research_file('data.txt')
    with pytest.raises(ValueError):await service.write_research_file('data.txt','overwrite')
    assert private.read_text()=='private'


@pytest.mark.asyncio
@pytest.mark.parametrize('mode',['local','online'])
async def test_attachment_normalizer_rejects_host_and_other_conversations(tmp_path,monkeypatch,mode):
    from backend.web_v2.chat_api._uploads import _normalize_uploaded_file
    from mcp_tools.common import common_utils
    monkeypatch.setenv('WEBUI_V2_MODE',mode)
    monkeypatch.setattr(common_utils,'get_web_v2_root_dir',lambda:tmp_path)
    work=tmp_path/'sessions'/'a';work.mkdir(parents=True)
    peer=tmp_path/'sessions'/'b';peer.mkdir();(peer/'data.csv').write_text('peer')
    host=tmp_path/'host-private';host.write_text('host')
    (work/'input.csv').write_text('owned')
    (work/'link.csv').symlink_to(host)
    for path in [str(host),str(peer/'data.csv'),'sessions/b/data.csv',str(work/'link.csv')]:
        assert await _normalize_uploaded_file(path,str(work),[],'same-user') is None
    files=[]
    result=await _normalize_uploaded_file('sessions/a/input.csv',str(work),files,'same-user')
    assert Path(result).read_text()=='owned' and files==[result]


@pytest.mark.asyncio
async def test_preview_keeps_authorized_descriptor_after_path_swap(tmp_path,monkeypatch):
    from backend.web_v2.chat_api import session_files
    work=tmp_path/'w';work.mkdir();p=work/'file.txt';p.write_text('owned')
    private=tmp_path/'private';private.write_text('secret')
    async def roots(*args):return {'agent':work}
    monkeypatch.setattr(session_files,'roots',roots)
    response=await session_files.content('session','agent/file.txt',None)
    p.unlink();p.symlink_to(private)
    body=b''.join([part async for part in response.body_iterator])
    assert body==b'owned'
    assert 'attachment;' in response.headers['content-disposition']


@pytest.mark.asyncio
async def test_local_library_download_does_not_reopen_path(tmp_path,monkeypatch):
    from backend.web_v2 import workspace_api
    work=tmp_path/'w';work.mkdir();p=work/'file.txt';p.write_text('owned')
    private=tmp_path/'private';private.write_text('secret')
    async def resolve(*args):return p,'agent/file.txt'
    monkeypatch.setattr(workspace_api,'_resolve_download',resolve)
    monkeypatch.setattr(workspace_api,'_load_workspace_index',lambda:{})
    response=await workspace_api.workspace_file_content(None,'agent/file.txt')
    p.unlink();p.symlink_to(private)
    assert b''.join([part async for part in response.body_iterator])==b'owned'


def test_valid_archive_contains_only_chosen_files(tmp_path):
    (tmp_path/'result.csv').write_text('x,y\n1,2')
    with archive_files({'agent':tmp_path},['agent/result.csv']) as stream, zipfile.ZipFile(stream) as archive:
        assert archive.namelist()==['agent/result.csv']
        assert archive.read('agent/result.csv')==b'x,y\n1,2'
