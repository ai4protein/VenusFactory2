"""Explicit post-provisioning acceptance test: real Cube microVMs, no cloud API."""
import asyncio
import json
import os

import pytest

pytestmark=pytest.mark.skipif(os.getenv('VENUS_CUBE_LIVE_TEST')!='1',reason='Local CubeSandbox service has not been provisioned for this test')


@pytest.mark.asyncio
async def test_real_cube_mount_persistence_and_deletion(tmp_path):
    from cubesandbox import Config,Sandbox
    from agent.harness.session_cube import configuration
    from agent.harness.session_sandbox import backend,execute
    assert backend()=='cubesandbox','Live test must not use a substitute backend'
    work=tmp_path/'session';work.mkdir();(work/'input.txt').write_text('session-fixture')
    (work/'probe.py').write_text('import os,json\nfrom pathlib import Path\nassert os.getcwd()=="/workspace"\nassert Path("input.txt").read_text()=="session-fixture"\nassert not any(k in os.environ for k in ["CUBE_API_KEY","E2B_API_KEY","OPENROUTER_API_KEY","DMXAPI_API_KEY"])\nassert not Path('+repr(str(work))+').exists()\nPath("proof.json").write_text(json.dumps({"cwd":os.getcwd(),"verified":True}))\n')
    first=await execute(work,tmp_path/'control','python','probe.py',30)
    assert first['exit_code']==0 and first['destroyed'],first
    assert json.loads((work/'proof.json').read_text())=={'cwd':'/workspace','verified':True}
    second=await execute(work,tmp_path/'control','shell','test -f proof.json && pwd',10)
    assert second['exit_code']==0 and second['stdout'].strip()=='/workspace'
    assert first['sandbox_id']!=second['sandbox_id'] and second['destroyed']
    options,_=configuration()
    active=await asyncio.to_thread(Sandbox.list,config=Config(**options))
    active_ids={item.get('sandboxID') or item.get('sandbox_id') for item in active}
    assert not {first['sandbox_id'],second['sandbox_id']} & active_ids
