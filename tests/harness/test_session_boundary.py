import asyncio
import io
import json
import os
from pathlib import Path
import tarfile

import pytest

from agent.harness.session_boundary import bind_arguments, public_value, session_path, session_tools
from agent.harness.session_sandbox import execute, backend, SandboxUnavailable
from agent.harness.tool_response_names import normalize


@pytest.mark.parametrize('value',['../private','/etc/passwd','/proc/self/environ','/home/another/file','/workspace/../private','.env','a/.ssh/key','file\\..\\secret'])
def test_paths_never_leave_session(tmp_path,value):
    with pytest.raises(ValueError):session_path(tmp_path,value)


def test_symlink_and_nested_tool_arguments_cannot_escape(tmp_path):
    work=tmp_path/'session';work.mkdir();(tmp_path/'private').write_text('secret')
    (work/'link').symlink_to(tmp_path/'private')
    for args in [{'input_file':'link'},{'inputs':{'pdb_path':'../private'}},{'config_path':'/etc/passwd'}]:
        with pytest.raises(ValueError):bind_arguments('calculate_sasa_from_pdb',args,work)
    assert bind_arguments('download_rcsb_structure_by_pdb_id',{'pdb_id':'1CRN','out_dir':'structures'},work)=={'pdb_id':'1CRN','out_dir':str(work/'structures')}
    assert bind_arguments('read_fasta',{'input_file':'/workspace/input.fasta'},work)['input_file']==str(work/'input.fasta')


def test_results_are_virtualized_and_credentials_removed(tmp_path):
    text=public_value({'artifact':str(tmp_path/'result.csv'),'diagnostic':'file /home/operator/.env API_KEY=fixture_secret_12345'},tmp_path)
    assert text['artifact']=='/workspace/result.csv'
    assert '/home/operator' not in text['diagnostic'] and 'fixture_secret' not in text['diagnostic']


@pytest.mark.asyncio
async def test_real_shell_cannot_read_host_or_other_session_or_credentials(tmp_path,monkeypatch):
    monkeypatch.setenv('PRIVATE_TEST_KEY','never_copy_to_child')
    a=tmp_path/'a';a.mkdir();b=tmp_path/'b';b.mkdir();(b/'secret').write_text('other_session')
    script='''import os,socket,json
assert 'PRIVATE_TEST_KEY' not in os.environ
assert not os.path.exists('/etc/passwd')
assert not os.path.exists('/home/tanyang')
assert not os.path.exists(%r)
s=socket.socket();s.settimeout(.2)
try:
 s.connect(('127.0.0.1',7863))
 raise AssertionError('host loopback reachable')
except OSError:pass
json.dump({'isolated':True},open('proof.json','w'))
''' % str(b/'secret')
    (a/'probe.py').write_text(script)
    value=await execute(a,tmp_path/'control','shell','python probe.py',10)
    assert value['exit_code']==0,value
    assert json.loads((a/'proof.json').read_text())=={'isolated':True}
    assert (b/'secret').read_text()=='other_session'


@pytest.mark.asyncio
async def test_session_tools_support_code_without_host_repl(tmp_path):
    tools={t.name:t for t in session_tools(tmp_path/'w',tmp_path/'control')}
    await tools['write_session_file'].ainvoke({'path':'calc.py','content':'print(sum([1,2,3]))'})
    value=await tools['run_session_python'].ainvoke({'path':'calc.py','timeout_seconds':10})
    assert value['exit_code']==0 and value['stdout'].strip()=='6'
    assert set(tools)=={'list_session_files','read_session_file','write_session_file','edit_session_file','run_session_python','run_session_shell'}
    with pytest.raises(ValueError):await tools['read_session_file'].ainvoke({'path':'../private'})


@pytest.mark.asyncio
async def test_timeout_kills_background_processes(tmp_path):
    work=tmp_path/'w';work.mkdir()
    (work/'run.py').write_text("import time\ntime.sleep(20)\nopen('late.txt','w').close()")
    with pytest.raises(TimeoutError):
        await execute(work,tmp_path/'control','python','run.py',1)
    await asyncio.sleep(3)
    assert not (work/'late.txt').exists()


def test_no_unsandboxed_fallback(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_SESSION_SANDBOX','user')
    with pytest.raises(SandboxUnavailable):backend()
    monkeypatch.setenv('VENUS_SESSION_SANDBOX','bwrap')
    monkeypatch.setattr('agent.harness.session_sandbox.shutil.which',lambda name:None)
    with pytest.raises(SandboxUnavailable):backend()


def test_legacy_approval_refuses_shell_and_cross_session_in_both_modes(tmp_path):
    from agent.kimi_security import decide
    for mode in ['local','online']:
        for name,args in [('Bash',{'command':'ls /'}),('Write',{'path':'x.py'}),('Skill',{}),('Read',{'path':'/etc/passwd'}),('Grep',{'path':'../'})]:
            assert not decide({'tool_name':name,'tool_input_display':args},session_dir=str(tmp_path),mode=mode).allowed


def test_exact_tool_alias_repair_and_unknown_name_refusal():
    tools=[{'function':{'name':'mcp__venusfactory__write_research_file'}}]
    obj={'choices':[{'message':{'tool_calls':[{'function':{'name':'write_research_file','arguments':'{}'}}]}}]}
    raw,repairs=normalize(json.dumps(obj).encode(),False,tools)
    assert json.loads(raw)['choices'][0]['message']['tool_calls'][0]['function']['name']==tools[0]['function']['name']
    assert len(repairs)==1
    obj['choices'][0]['message']['tool_calls'][0]['function']['name']='Bash'
    with pytest.raises(ValueError):normalize(json.dumps(obj).encode(),False,tools)


def test_fragmented_stream_names_are_repaired_without_changing_arguments():
    raw=b''.join(b'data: '+json.dumps({'choices':[{'index':0,'delta':{'tool_calls':[{'index':0,'function':{'name':name,'arguments':arg}}]}}]}).encode()+b'\n\n' for name,arg in [('write_','{"path":'),('research_file','"ok.py"}')])+b'data: [DONE]\n\n'
    out,_=normalize(raw,True,[{'function':{'name':'mcp__venusfactory__write_research_file'}}])
    rows=[json.loads(l[6:]) for l in out.splitlines() if l.startswith(b'data: {')]
    fns=[r['choices'][0]['delta']['tool_calls'][0]['function'] for r in rows]
    assert fns[0]['name']=='mcp__venusfactory__write_research_file' and 'name' not in fns[1]
    assert ''.join(f['arguments'] for f in fns)=='{"path":"ok.py"}'


def test_e2b_cannot_default_to_cloud(monkeypatch):
    from agent.harness.session_e2b import configuration
    monkeypatch.delenv('E2B_API_URL',raising=False)
    with pytest.raises(SandboxUnavailable):configuration()
    monkeypatch.setenv('E2B_API_URL','https://api.e2b.app')
    with pytest.raises(SandboxUnavailable):configuration()


@pytest.mark.parametrize('name,kind',[('../escape','file'),('/tmp/escape','file'),('bad','symlink'),('bad','hardlink')])
def test_e2b_import_never_trusts_archive_paths(tmp_path,name,kind):
    from agent.harness.session_e2b import import_outputs
    data=io.BytesIO()
    with tarfile.open(fileobj=data,mode='w') as t:
        info=tarfile.TarInfo(name);info.size=0
        if kind=='symlink':info.type=tarfile.SYMTYPE;info.linkname='/etc/passwd'
        if kind=='hardlink':info.type=tarfile.LNKTYPE;info.linkname='/etc/passwd'
        t.addfile(info)
    with pytest.raises(ValueError):import_outputs(data.getvalue(),tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize('function,kwargs',[
    ('_resolve_kimi_question_and_resume',{'answers_data':[]}),
    ('_resolve_kimi_approval_decision_and_resume',{'approved':True}),
])
async def test_obsolete_resume_cannot_contact_host_daemon(monkeypatch,function,kwargs):
    from fastapi import HTTPException
    from unittest.mock import AsyncMock
    from backend.web_v2.chat_api import interactive

    client=AsyncMock(side_effect=AssertionError('host daemon must not be contacted'))
    monkeypatch.setattr(interactive,'acquire_kimi_client_for_state',client)
    state={'history':[]}
    with pytest.raises(HTTPException) as error:
        await anext(getattr(interactive,function)(state,**kwargs))
    assert error.value.status_code==409
    assert state=={'history':[]}
    client.assert_not_awaited()
