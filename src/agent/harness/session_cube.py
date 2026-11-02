"""One local CubeSandbox microVM per MCP execution, mounting only its session."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import secrets
import shlex
import time
import uuid
from urllib.parse import urlsplit

from .session_sandbox import SandboxUnavailable, check_workspace, validate_execution


def configuration():
    endpoint=os.getenv('CUBE_API_URL','')
    parsed=urlsplit(endpoint)
    try:local=ipaddress.ip_address(parsed.hostname or '').is_loopback
    except ValueError:local=parsed.hostname=='localhost'
    if not local or parsed.scheme not in {'http','https'} or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise SandboxUnavailable('Configure a local CubeAPI endpoint')
    proxy=os.getenv('CUBE_PROXY_NODE_IP','127.0.0.1')
    try:local_proxy=ipaddress.ip_address(proxy).is_loopback
    except ValueError:local_proxy=False
    if not local_proxy:raise SandboxUnavailable('Configure a local CubeProxy endpoint')
    template=os.getenv('CUBE_TEMPLATE_ID','')
    node=os.getenv('CUBE_COMPUTE_NODE','')
    key=os.getenv('CUBE_API_KEY','')
    if not template or not node or not key:
        raise SandboxUnavailable('Configure the Cube template, local compute node and service key')
    port=int(os.getenv('CUBE_PROXY_PORT_HTTP','80'))
    if not 1<=port<=65535:raise SandboxUnavailable('Invalid CubeProxy port')
    return {'api_url':endpoint,'api_key':key,'template_id':template,'proxy_node_ip':proxy,
            'proxy_port':port,'sandbox_domain':'cube.app','request_timeout':15},node


async def _settle(task):
    """Drain SDK requests when cancellation arrives, retaining created VM handles."""
    interrupted=False
    while True:
        try:return await asyncio.shield(task),interrupted
        except asyncio.CancelledError:
            if task.cancelled():raise
            interrupted=True


async def _destroy(sandbox):
    from cubesandbox import SandboxNotFoundError
    for attempt in range(2):
        try:
            await asyncio.to_thread(sandbox.kill)
            return
        except SandboxNotFoundError:return
        except Exception:
            if attempt:raise
            await asyncio.sleep(.2)


async def execute(workspace, control, kind, payload, timeout, *, on_progress=None):
    validate_execution(kind,payload,timeout)
    online=os.getenv('VENUS_SCIENTIST_ONLINE')=='1' or os.getenv('WEBUI_V2_MODE')=='online'
    if online:timeout=min(timeout,60)
    check_workspace(workspace)
    options,node=configuration()
    try:from cubesandbox import Config, Sandbox
    except ImportError:raise SandboxUnavailable('Install requirements/optional/cube.txt') from None
    workspace=Path(workspace).resolve();control=Path(control);control.mkdir(parents=True,exist_ok=True)
    identity=uuid.uuid4().hex;ttl=timeout+90
    marker=workspace/('.cube-bind-'+identity);challenge=secrets.token_hex(24)
    with marker.open('x') as stream:stream.write(challenge)
    marker.chmod(0o600)
    lifecycle=control/(identity+'.lifecycle.json')
    record={'run_id':identity,'backend':'cubesandbox','idle_timeout_seconds':ttl,'events':[]}
    def phase(name,**details):
        record.update(stage=name,**details);record['events'].append({'stage':name,'at':time.time()})
        temp=lifecycle.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.chmod(0o600);temp.replace(lifecycle)
    def progress(name):
        if on_progress:on_progress({'status':'running','phase':name,'backend':'cubesandbox','run_id':identity})
    sbx=None;creating=None;pending=None;failure=None;cancelled=False;value=None
    try:
        phase('creating');progress('creating')
        creating=asyncio.create_task(asyncio.to_thread(Sandbox.create,config=Config(**options),timeout=ttl,
            distribution_scope=[node],allow_internet_access=False,
            network={'deny_out':['0.0.0.0/0','::/0'],'allow_public_traffic':False},
            lifecycle={'on_timeout':'kill','auto_resume':False},
            metadata={'venus_run':identity,'venus_session':hashlib.sha256(str(workspace).encode()).hexdigest(),
                      'host-mount':json.dumps([{'hostPath':str(workspace),'mountPath':'/workspace'}])}))
        sbx=await asyncio.shield(creating)
        phase('created',sandbox_id=sbx.sandbox_id)
        async with asyncio.timeout(timeout+45):
            # A node pin alone is insufficient: prove that the supplied session
            # actually appeared, with the template's unprivileged user identity.
            mounted=await asyncio.to_thread(sbx.files.read,'/workspace/'+marker.name,user='user')
            if mounted!=challenge:raise SandboxUnavailable('Cube session mount could not be verified')
            runner='/tmp/venus-'+identity+'.py'
            await asyncio.to_thread(sbx.files.write,runner,Path(__file__).with_name('session_command_worker.py').read_text(),user='root')
            output='/tmp/venus-'+identity
            cmd='env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/home/user OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 '
            if online:cmd+='VENUS_SCIENTIST_ONLINE=1 '
            cmd+='python3 -I -u '+shlex.join([runner,kind,str(timeout),payload])+' > '+output+'.out 2> '+output+'.err'
            phase('executing');progress('executing')
            pending=asyncio.create_task(asyncio.to_thread(sbx.commands.run,cmd,cwd='/workspace',user='user',timeout=timeout))
            while not pending.done():
                check_workspace(workspace)
                await asyncio.wait({pending},timeout=.25)
            result=await pending
            out=await asyncio.to_thread(sbx.commands.run,'/usr/bin/tail -c 12000 '+output+'.out',user='user',timeout=5)
            err=await asyncio.to_thread(sbx.commands.run,'/usr/bin/tail -c 6000 '+output+'.err',user='user',timeout=5)
            value={'run_id':identity,'sandbox_id':sbx.sandbox_id,'backend':'cubesandbox',
                   'exit_code':result.exit_code,'stdout':out.stdout,'stderr':err.stdout}
    except BaseException as exc:
        failure=exc
    # Do not cancel the create thread and lose the only handle to a new VM.
    if sbx is None and creating is not None:
        try:sbx,cancelled=await _settle(creating)
        except BaseException as exc:
            if failure is None:failure=exc
    if sbx is not None:
        try:
            with contextlib.suppress(OSError):phase('destroying',sandbox_id=sbx.sandbox_id)
            _,interrupted=await _settle(asyncio.create_task(_destroy(sbx)))
            cancelled=cancelled or interrupted
            phase('destroyed',destroyed=True)
            if value is not None:value['destroyed']=True
        except BaseException as exc:
            with contextlib.suppress(OSError):phase('destroy_failed',destroyed=False,error_type=type(exc).__name__)
            if failure is None:failure=SandboxUnavailable('CubeSandbox deletion was not confirmed; server idle expiry remains configured')
        finally:
            if pending is not None:
                try:
                    _,interrupted=await _settle(pending);cancelled=cancelled or interrupted
                except BaseException:pass
            with contextlib.suppress(Exception):await asyncio.to_thread(sbx.close)
    else:
        with contextlib.suppress(OSError):phase('create_unconfirmed',destroyed=False)
    marker.unlink(missing_ok=True)
    if cancelled:raise asyncio.CancelledError()
    if failure is not None:raise failure
    check_workspace(workspace)
    if on_progress:on_progress({**value,'status':'completed' if value['exit_code']==0 else 'failed','stdout_tail':value['stdout'],'stderr_tail':value['stderr']})
    return value
