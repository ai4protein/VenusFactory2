import json
from pathlib import Path

import pytest

from mcp_tools.tools.research import interproscan as ipr


@pytest.fixture
def tickets(tmp_path,monkeypatch):
    monkeypatch.setenv('VENUS_JOB_QUEUE_ROOT',str(tmp_path/'queue'))
    monkeypatch.setenv('VENUS_QUEUE_OWNER','annotation-owner')
    monkeypatch.setattr(ipr,'configuration',lambda:'science@example.org')
    return ipr.Tickets()


@pytest.mark.asyncio
async def test_interproscan_submit_once_poll_and_owner_isolation(tickets,monkeypatch):
    from mcp_tools.tools.research import services
    calls=[];clock=[1000.0]
    monkeypatch.setattr(ipr.time,'time',lambda:clock[0])
    async def request(method,url,**kwargs):
        calls.append((method,url,kwargs))
        if method=='POST':return b'iprscan5-test-12345678'
        if '/status/' in url:return b'FINISHED'
        return b'query\tmd5\t20\tPfam\tPF00001\tDomain\t1\t20\t1e-10\tT\t2026-11-02\tIPR00001\tDomain name\tGO:0000001\t-\n'
    monkeypatch.setattr(services,'request',request)
    row={};seq='ACDEFGHIKLMNPQRSTVWYAC'
    assert await ipr.run({'sequence':seq,'submit_only':True},row,lambda:None) is None
    identity=row['job_id'];assert len(identity)==32 and row['status']=='pending'
    assert await ipr.run({'sequence':seq,'submit_only':True},{},lambda:None) is None
    assert len(calls)==1
    # Early polling does not hammer EBI and never resubmits.
    assert await ipr.run({'remote_job_id':identity,'poll_attempts':1},{},lambda:None) is None
    assert len(calls)==1
    monkeypatch.setenv('VENUS_QUEUE_OWNER','another-owner')
    with pytest.raises(PermissionError):await ipr.run({'remote_job_id':identity},{},lambda:None)
    monkeypatch.setenv('VENUS_QUEUE_OWNER','annotation-owner');clock[0]+=16
    result=await ipr.run({'remote_job_id':identity,'poll_attempts':1},{},lambda:None)
    assert result['count']==1 and result['hits'][0]['signature']=='PF00001'
    assert result['hits'][0]['go_terms']=='GO:0000001'
    assert await ipr.run({'remote_job_id':identity},{},lambda:None)==result
    assert len(calls)==3


@pytest.mark.asyncio
async def test_unknown_submission_never_silently_resubmits(tickets,monkeypatch):
    from mcp_tools.tools.research import services
    async def fail(*args,**kwargs):raise TimeoutError('fixture')
    monkeypatch.setattr(services,'request',fail)
    context={'sequence':'ACDEFGHIKLMNPQRSTVWYAC','submit_only':True}
    with pytest.raises(TimeoutError):await ipr.run(context,{},lambda:None)
    with pytest.raises(services.ServiceError,match='not resubmitted'):await ipr.run(context,{},lambda:None)


def test_ticket_limit_and_identifier_validation(tickets):
    for i in range(4):tickets.get_or_create(sequence='A'*20+str(i))
    with pytest.raises(ValueError,match='busy'):tickets.get_or_create(sequence='new')
    with pytest.raises(ValueError):tickets.get_or_create(identity='../../etc/passwd')
    with pytest.raises(ValueError):ipr.parse_tsv(b'incomplete\trow')


def test_s4pred_parse_and_no_user_command_arguments(tmp_path,monkeypatch):
    import mcp_tools.models.pretrained.s4pred as s4pred
    from mcp_tools.models.pretrained.design_docking import service
    from mcp_tools.tools import runtime_tool_policy
    seq='ACDEFGHIKLMN'
    monkeypatch.setenv('TEMP_OUTPUTS_DIR',str(tmp_path/'output'))
    monkeypatch.setattr(runtime_tool_policy,'assert_tool_call_allowed',lambda *a:None)
    monkeypatch.setattr(runtime_tool_policy,'is_online_mode',lambda:True)
    monkeypatch.setattr(s4pred,'runtime',lambda:{'python':'/runtime/bin/python','root':'/runtime/S4PRED'})
    def execute(argv,**kwargs):
        out=Path(argv[argv.index('--outdir')+1]);source=Path(argv[2])
        assert source.read_text()=='>query\n'+seq+'\n'
        assert argv[argv.index('--device')+1]=='cpu'
        assert kwargs['env'].get('CUDA_VISIBLE_DEVICES')==''
        (out/'s4_out_0.ss2').write_text('# S4PRED\n'+'\n'.join(f'{i+1} {aa} C 0.8 0.1 0.1' for i,aa in enumerate(seq)))
    monkeypatch.setattr(service,'execute_command',execute)
    result=s4pred.predict(seq)
    assert result['count']==len(seq) and result['secondary_structure']=='C'*len(seq)
    assert all(Path(p).exists() for p in result['artifacts'])
    with pytest.raises(ValueError):s4pred.predict('A'*2001)
    with pytest.raises(ValueError):s4pred.predict('A;cat /etc/passwd')


def test_plaza_and_mcp_annotation_contracts(monkeypatch):
    from backend.web_v2.plaza_tools import card,registered
    import mcp_tools.models.pretrained.s4pred as s4pred
    from mcp_tools.scheduler.catalog import classify
    monkeypatch.setattr(s4pred,'runtime',lambda:{})
    monkeypatch.setattr(ipr,'configuration',lambda:'science@example.org')
    for name in ['predict_s4pred','query_interproscan']:
        model=card(name)
        assert model['input_kind']=='table'
        assert model['contract']['properties']=={} and model['contract']['required']==[]
        assert registered(name).name==name
    assert classify('predict_s4pred')['resource']=='cpu'
    assert classify('query_interproscan')['resource'].startswith('api:')
