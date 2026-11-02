"""Opt-in self-hosted E2B execution. Never defaults to E2B Cloud.

The E2B API does not bind-mount an arbitrary host directory. We transfer only
regular session files and validate the returned archive before publishing them.
Each execution gets a fresh VM; the persistent state is the session directory.
"""
from __future__ import annotations
import asyncio
import io
import ipaddress
import os
from pathlib import Path, PurePosixPath
import shlex
import tarfile
import uuid
from urllib.parse import urlsplit

from .session_sandbox import SandboxUnavailable, check_workspace


def configuration():
    opts={}
    for key,env in [('api_url','E2B_API_URL'),('sandbox_url','E2B_SANDBOX_URL')]:
        value=os.getenv(env,'');parsed=urlsplit(value)
        try: local=ipaddress.ip_address(parsed.hostname or '').is_loopback
        except ValueError: local=parsed.hostname=='localhost'
        if not local or parsed.scheme not in {'http','https'} or parsed.username or parsed.password:
            raise SandboxUnavailable('Configure local E2B API and sandbox proxy URLs; cloud fallback is disabled')
        opts[key]=value
    key=os.getenv('E2B_API_KEY','')
    template=os.getenv('VENUS_E2B_TEMPLATE','')
    if not key or not template:raise SandboxUnavailable('Configure the self-hosted E2B key and scientific template')
    return {**opts,'api_key':key,'domain':'localhost','request_timeout':30,'retries':0},template


def import_outputs(data, workspace):
    from .session_boundary import session_path
    if len(data)>200*1024**2:raise ValueError('Sandbox output archive is too large')
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as archive:
        members=archive.getmembers()
        if len(members)>10000:raise ValueError('Sandbox file limit reached')
        total=0;seen=set();validated=[]
        for item in members:
            name=item.name.removeprefix('./')
            if name in {'','.'}:continue
            if not item.isdir() and not item.isfile():raise ValueError('Sandbox links and special files cannot be imported')
            if name in seen:raise ValueError('Duplicate sandbox output')
            seen.add(name)
            dest=Path(session_path(workspace,name))
            total+=item.size
            if item.size>100*1024**2 or total>200*1024**2:raise ValueError('Sandbox output limit reached')
            validated.append((item,dest))
        for item,dest in validated:
            if item.isdir():dest.mkdir(parents=True,exist_ok=True);continue
            dest.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(item) as source:
                stage=dest.with_name(dest.name+'.'+uuid.uuid4().hex+'.tmp')
                with stage.open('xb') as out:out.write(source.read())
                stage.chmod(0o600);stage.replace(dest)


async def execute(workspace, control, kind, payload, timeout, *, on_progress=None):
    opts,template=configuration()
    try:from e2b import AsyncSandbox
    except ImportError:raise SandboxUnavailable('Install the optional self-hosted E2B SDK') from None
    sbx=None
    identity=uuid.uuid4().hex
    # The local template must include Python, scientific packages and /workspace
    # owned by the non-root user. No keys are passed into its command environment.
    try:
        async with asyncio.timeout(timeout+90):
            sbx=await AsyncSandbox.create(template=template,timeout=timeout+60,allow_internet_access=False,
                network={'deny_out':['0.0.0.0/0','::/0'],'allow_public_traffic':False},
                metadata={'venus_session':__import__('hashlib').sha256(str(Path(workspace).resolve()).encode()).hexdigest()},**opts)
            check_workspace(workspace)
            for p in Path(workspace).rglob('*'):
                if p.is_file() and not any(x.startswith('.') for x in p.relative_to(workspace).parts):
                    await sbx.files.write('/workspace/'+p.relative_to(workspace).as_posix(),p.read_bytes())
            await sbx.files.write('/tmp/venus-runner.py',Path(__file__).with_name('session_command_worker.py').read_text())
            cmd='env -i PATH=/usr/local/bin:/usr/bin:/bin HOME=/home/user OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VENUS_SCIENTIST_ONLINE=1 python3 -I -u /tmp/venus-runner.py '+shlex.join([kind,str(timeout),payload])
            result=await sbx.commands.run(cmd,cwd='/workspace',user='user',timeout=timeout)
            # Export a bounded snapshot, never extract a sandbox-authored path on host.
            await sbx.commands.run('tar --format=ustar -cf /tmp/venus-output.tar -C /workspace .',user='user',timeout=15)
            info=await sbx.files.get_info('/tmp/venus-output.tar')
            if info.size>200*1024**2:raise ValueError('Sandbox output limit reached')
            data=await sbx.files.read('/tmp/venus-output.tar',format='bytes')
            import_outputs(bytes(data),workspace)
            value={'run_id':identity,'backend':'e2b','exit_code':result.exit_code,'stdout':result.stdout[-12000:],'stderr':result.stderr[-6000:]}
            if on_progress:on_progress({**value,'status':'completed' if result.exit_code==0 else 'failed','stdout_tail':value['stdout'],'stderr_tail':value['stderr']})
            return value
    finally:
        if sbx is not None:await asyncio.shield(sbx.kill())
