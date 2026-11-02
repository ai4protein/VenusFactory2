"""Cube SDK lifecycle contract tests; these do not claim a running microVM."""
import asyncio
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from agent.harness import session_cube
from agent.harness.session_sandbox import SandboxUnavailable, execute


@pytest.fixture
def sdk(tmp_path,monkeypatch):
    from cubesandbox import Sandbox
    for key,value in {'VENUS_SESSION_SANDBOX':'cubesandbox','CUBE_API_URL':'http://127.0.0.1:3000',
                      'CUBE_TEMPLATE_ID':'reviewed-test','CUBE_COMPUTE_NODE':'local-node','CUBE_API_KEY':'fixture-service-key',
                      'CUBE_PROXY_NODE_IP':'127.0.0.1','CUBE_PROXY_PORT_HTTP':'80'}.items():monkeypatch.setenv(key,value)
    for key in ['VENUS_SCIENTIST_ONLINE','WEBUI_V2_MODE']:monkeypatch.delenv(key,raising=False)
    workspace=tmp_path/'session';workspace.mkdir();control=tmp_path/'control'
    state=SimpleNamespace(workspace=workspace,control=control,instances=[],options=[],create_entered=threading.Event(),
        run_entered=threading.Event(),create_gate=None,run_gate=None,bad_mount=False,error=None,exit_code=0,kill_failures=0)

    class Files:
        def read(self,path,*,user):
            assert user=='user'
            return 'wrong-session' if state.bad_mount else (workspace/Path(path).name).read_text()
        def write(self,path,data,*,user):
            assert path.startswith('/tmp/venus-') and user=='root'
            assert 'fixture-service-key' not in data

    class Commands:
        def run(self,command,**kwargs):
            assert kwargs['user']=='user'
            assert 'fixture-service-key' not in command
            if '/usr/bin/tail' in command:
                return SimpleNamespace(stdout='verified\n' if command.endswith('.out') else '',exit_code=0)
            assert kwargs['cwd']=='/workspace' and 'env -i ' in command
            state.run_entered.set()
            if state.run_gate:assert state.run_gate.wait(5)
            if state.error:raise state.error
            return SimpleNamespace(exit_code=state.exit_code,stdout='',stderr='')

    class Instance:
        def __init__(self):
            self.sandbox_id='test-vm-'+str(len(state.instances));self.files=Files();self.commands=Commands()
            self.kills=0;self.closed=False;state.instances.append(self)
        def kill(self):
            self.kills+=1
            if state.run_gate:state.run_gate.set()
            if self.kills<=state.kill_failures:raise ConnectionError('fixture deletion failure')
        def close(self):self.closed=True

    def create(**kwargs):
        state.options.append(kwargs);state.create_entered.set()
        if state.create_gate:assert state.create_gate.wait(5)
        return Instance()
    monkeypatch.setattr(Sandbox,'create',create)
    return state


@pytest.mark.asyncio
async def test_fresh_vm_exact_session_mount_and_destroy_before_return(sdk):
    first=await execute(sdk.workspace,sdk.control,'shell','pwd',10)
    second=await execute(sdk.workspace,sdk.control,'shell','pwd',10)
    assert first['sandbox_id']!=second['sandbox_id']
    assert all(result['backend']=='cubesandbox' and result['destroyed'] for result in [first,second])
    assert all(vm.kills==1 and vm.closed for vm in sdk.instances)
    for options in sdk.options:
        assert json.loads(options['metadata']['host-mount'])==[{'hostPath':str(sdk.workspace),'mountPath':'/workspace'}]
        assert options['distribution_scope']==['local-node']
        assert options['lifecycle']=={'on_timeout':'kill','auto_resume':False}
        assert not options['allow_internet_access'] and not options['network']['allow_public_traffic']
        assert options['network']['deny_out']==['0.0.0.0/0','::/0']
        assert 'env_vars' not in options and 'envs' not in options
    assert not list(sdk.workspace.glob('.cube-bind-*'))
    assert all(json.loads(p.read_text())['stage']=='destroyed' for p in sdk.control.glob('*.lifecycle.json'))


@pytest.mark.asyncio
@pytest.mark.parametrize('problem',['program_failed','transport','timeout','wrong_mount'])
async def test_failed_work_still_destroys_vm(sdk,problem):
    if problem=='program_failed':sdk.exit_code=1
    elif problem=='transport':sdk.error=ConnectionError('fixture')
    elif problem=='timeout':sdk.error=TimeoutError('fixture')
    else:sdk.bad_mount=True
    if problem=='program_failed':
        result=await execute(sdk.workspace,sdk.control,'shell','false',10)
        assert result['exit_code']==1 and result['destroyed']
    else:
        with pytest.raises((ConnectionError,TimeoutError,SandboxUnavailable)):
            await execute(sdk.workspace,sdk.control,'shell','pwd',10)
    assert sdk.instances[0].kills==1 and sdk.instances[0].closed
    if problem=='wrong_mount':assert not sdk.run_entered.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('stage',['create','execute'])
async def test_cancellation_retains_handle_and_destroys_vm(sdk,stage):
    gate=threading.Event()
    if stage=='create':sdk.create_gate=gate;started=sdk.create_entered
    else:sdk.run_gate=gate;started=sdk.run_entered
    task=asyncio.create_task(execute(sdk.workspace,sdk.control,'shell','sleep 20',10))
    assert await asyncio.to_thread(started.wait,3)
    task.cancel()
    if stage=='create':gate.set()
    with pytest.raises(asyncio.CancelledError):await asyncio.wait_for(task,5)
    assert sdk.instances[0].kills==1 and sdk.instances[0].closed
    assert not list(sdk.workspace.glob('.cube-bind-*'))


@pytest.mark.asyncio
async def test_delete_failure_is_not_reported_as_success(sdk):
    sdk.kill_failures=2
    with pytest.raises(SandboxUnavailable,match='deletion was not confirmed'):
        await execute(sdk.workspace,sdk.control,'shell','pwd',10)
    assert sdk.instances[0].kills==2
    state=json.loads(next(sdk.control.glob('*.lifecycle.json')).read_text())
    assert state['stage']=='destroy_failed' and state['destroyed'] is False


@pytest.mark.asyncio
async def test_online_timeout_applies_before_cube_dispatch(sdk,monkeypatch):
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    await execute(sdk.workspace,sdk.control,'shell','pwd',300)
    assert sdk.options[0]['timeout']==150


@pytest.mark.asyncio
@pytest.mark.parametrize('timeout',[0,301,True])
async def test_bad_budgets_never_create_vm(sdk,timeout):
    with pytest.raises(ValueError):await execute(sdk.workspace,sdk.control,'shell','pwd',timeout)
    assert not sdk.instances


@pytest.mark.parametrize('key,value',[('CUBE_API_URL','https://remote.example'),('CUBE_API_URL','http://127.0.0.1:3000/?key=x'),('CUBE_API_KEY',''),('CUBE_COMPUTE_NODE',''),('CUBE_PROXY_NODE_IP','10.0.0.1')])
def test_configuration_cannot_silently_use_remote_or_unbound_service(sdk,monkeypatch,key,value):
    monkeypatch.setenv(key,value)
    with pytest.raises(SandboxUnavailable):session_cube.configuration()
