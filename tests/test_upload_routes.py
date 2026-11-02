"""Exercise the actual upload handlers without starting AI/queue services."""
import ast
import asyncio
import io
import os
from pathlib import Path
from types import SimpleNamespace
from typing import List
import uuid

import pytest
from fastapi import Depends, File, HTTPException, Query, Request, UploadFile
from starlette.datastructures import UploadFile as Uploaded

from backend.web_v2 import upload_limits
from backend.web_v2.research_files import save_inputs

ROOT=Path(__file__).resolve().parents[1]
HANDLERS=[
    ('chat_api/attachments.py','upload_attachments','agent'),
    ('chat_api/science_goals.py','upload_research_inputs','scientist'),
    ('report_api.py','upload','report'),
    ('model_platform_api.py','upload','platform'),
    ('custom_model_api.py','upload_custom_model_file','custom'),
    ('custom_model_api.py','upload_predict_input_file','predict'),
    ('advanced_tools_api.py','upload_advanced_file','advanced'),
    ('quick_tools_api.py','upload_quick_tool_file','quick'),
    ('workspace_api.py','upload_workspace_file','workspace'),
    ('workspace_api.py','replace_workspace_file','replace'),
]


@pytest.mark.parametrize('path,name,kind',HANDLERS)
def test_every_file_upload_rejects_online_oversize(tmp_path,monkeypatch,path,name,kind):
    monkeypatch.setenv('WEBUI_V2_MODE','online')
    folder=tmp_path/'uploads';folder.mkdir(); existing=folder/'existing.csv';existing.write_bytes(b'original')
    state={'agent_session_dir':str(folder),'temp_files':[]}
    async def get(*args):return state
    async def call(fn,*args):return fn(*args)
    async def no_op(*args):return None
    def owner(*args):return 'owner'
    service=SimpleNamespace(session_folder=lambda sid:folder,upload=lambda *args:None)
    def new_path(*args):
        if kind=='quick':return 'run',folder/'input.pdb','input.pdb'
        return 'run',folder/'input.pdb'
    scope=dict(os=os,Path=Path,uuid=uuid,List=List,UploadFile=UploadFile,File=File,Request=Request,Depends=Depends,Query=Query,
        HTTPException=HTTPException,_get_session_or_404=get,_assert_session_access=lambda *a:None,_record_access_event=lambda *a:None,
        make_web_v2_upload_name=lambda i,name:name,to_web_v2_public_path=str,access=get,service=service,call=call,
        _queue_owner=owner,owner=owner,build_run_id_utc=lambda:'run',get_web_v2_area_dir=lambda *a,**k:folder,
        JobRecords=lambda:SimpleNamespace(grant=lambda *a:None),_new_upload_path=new_path,
        _owner_key_for_request=owner,_ALLOWED_UPLOAD_SUFFIX={'.csv','.pdb'},
        resolve_web_v2_client_path=lambda *a,**k:existing,_assert_user_upload_bucket=lambda *a:None,
        _load_workspace_index=lambda:{},_assert_online_workspace_access=no_op,_bucket_from_relative=lambda path:'user_upload')
    tree=ast.parse((ROOT/'backend/web_v2'/path).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.AsyncFunctionDef) and n.name==name);node.decorator_list=[]
    exec(compile(ast.Module([node],type_ignores=[]),path,'exec'),scope)
    file=Uploaded(io.BytesIO(b'no read needed'),filename='input.pdb' if kind in {'quick','advanced'} else 'input.csv',size=upload_limits.ONLINE_FILE_BYTES+1)
    if kind in {'agent','scientist'}:args={'session_id':'owned','request':None,'files':[file]}
    elif kind in {'custom','predict'}:args={'file':file,'owner':'owner'}
    elif kind in {'platform','report'}:args={'file':file,'who':'owner'}
    elif kind=='replace':args={'file':file,'request':None,'storage_path':'existing.csv'}
    elif kind=='workspace':args={'file':file,'request':None}
    else:args={'file':file}
    with pytest.raises(HTTPException) as error:asyncio.run(scope[name](**args))
    assert error.value.status_code==413
    assert '10 MB' in error.value.detail
    assert existing.read_bytes()==b'original'
    assert list(folder.iterdir()) == [existing]
