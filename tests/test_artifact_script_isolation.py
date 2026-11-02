import io
import importlib

import pytest
from starlette.requests import Request


def assert_isolated(response):
    policy=response.headers['content-security-policy']
    assert policy.split(';',1)[0]=='sandbox'
    assert 'allow-scripts' not in policy and 'allow-same-origin' not in policy
    assert "default-src 'none'" in policy and "form-action 'none'" in policy
    assert response.headers['x-content-type-options']=='nosniff'
    assert response.headers['referrer-policy']=='no-referrer'


@pytest.mark.asyncio
@pytest.mark.parametrize('module,handler',[
    ('backend.web_v2.quick_tools_api','download_quick_tool_result'),
    ('backend.web_v2.advanced_tools_api','download_advanced_result'),
])
async def test_legacy_inline_results_are_sandboxed_on_direct_navigation(tmp_path,monkeypatch,module,handler):
    api=importlib.import_module(module)
    artifact=tmp_path/'report.html';artifact.write_text('<script>fetch("/api/admin/env")</script>')
    monkeypatch.setattr(api,'resolve_web_v2_client_path',lambda *a,**k:artifact)
    monkeypatch.setattr(api,'ensure_within_roots',lambda *a:True)
    monkeypatch.setattr(api,'guard_artifact',lambda *a:None)
    response=await getattr(api,handler)(Request({'type':'http','headers':[]}),str(artifact),True)
    assert response.media_type.startswith('text/html')
    assert_isolated(response)


@pytest.mark.asyncio
@pytest.mark.parametrize('suffix',['.html','.svg'])
async def test_generic_inline_files_have_server_enforced_script_isolation(tmp_path,monkeypatch,suffix):
    from backend import api_server
    monkeypatch.setenv('TEMP_OUTPUTS_DIR',str(tmp_path))
    monkeypatch.setattr(api_server,'guard_artifact',lambda *a:None)
    artifact=tmp_path/('untrusted'+suffix);artifact.write_text('<script>alert(1)</script>')
    response=await api_server.get_inline_file(Request({'type':'http','headers':[]}),str(artifact))
    assert_isolated(response)


@pytest.mark.asyncio
async def test_descriptor_download_retains_attachment_and_isolation():
    from backend.web_v2.scoped_downloads import download
    response=download(io.BytesIO(b'fixture'),'report.html',media_type='text/html')
    assert_isolated(response)
    assert response.headers['content-disposition'].startswith('attachment;')
    assert b''.join([part async for part in response.body_iterator])==b'fixture'
