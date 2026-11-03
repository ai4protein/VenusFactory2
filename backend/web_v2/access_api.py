"""Quota self-service and authenticated administration. No bootstrap HTTP route."""
import sqlite3
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Query
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictBool
from starlette.concurrency import run_in_threadpool
from backend.web_v2.job_access import require_same_origin
from backend.web_v2.online_access import AccessStore, ADMIN_COOKIE, client_ip, fingerprint, identities, status, secure_request

router = APIRouter(prefix='/api')


def admin_cookie():
    from mcp_tools.tools.capability_policy import mode
    return ADMIN_COOKIE + '_' + mode()


async def call(fn, *args, **kwargs):
    try: return await run_in_threadpool(fn,*args,**kwargs)
    except sqlite3.IntegrityError: raise HTTPException(409,'This email address already exists') from None
    except ValueError as exc: raise HTTPException(422,str(exc)) from None


def private(request: Request, response: Response):
    require_same_origin(request)
    response.headers['Cache-Control']='no-store'
    response.headers['X-Robots-Tag']='noindex, nofollow'
    if request.method not in {'GET','HEAD'} and request.headers.get('x-venus-access')!='1':
        raise HTTPException(403,'Use the administration or quota form to submit changes')


def admin(request: Request, response: Response):
    private(request,response)
    return AccessStore().authenticate(request.cookies.get(admin_cookie(),'') or request.cookies.get(ADMIN_COOKIE,''))


class Credentials(BaseModel):
    model_config=ConfigDict(extra='forbid')
    email: str=Field(min_length=3,max_length=254)
    key: str=Field(min_length=1,max_length=256)


class Login(BaseModel):
    model_config=ConfigDict(extra='forbid')
    email: str=Field(min_length=3,max_length=254)
    password: str=Field(min_length=1,max_length=256)


class Administrator(BaseModel):
    model_config=ConfigDict(extra='forbid')
    email: str=Field(min_length=3,max_length=254)
    password: str=Field(default='',max_length=256)
    active: bool=True


class Grant(BaseModel):
    model_config=ConfigDict(extra='forbid')
    email: str=Field(min_length=3,max_length=254)
    extra: StrictInt=Field(default=3,ge=0,le=10000)
    extra_predictions: StrictInt=Field(default=0,ge=0,le=10000)
    active: bool=True
    expires: float | None=Field(default=None,gt=0)
    name: str=Field(default='',max_length=200)
    institution: str=Field(default='',max_length=300)
    contact: str=Field(default='',max_length=200)
    note: str=Field(default='',max_length=2000)


@router.get('/quota',dependencies=[Depends(private)])
async def quota(request: Request):
    result = await call(status,request)
    if result['enforced']:
        from backend.web_v2.compute_access import ComputeStore
        result['compute'] = await call(ComputeStore().status,actor=identities(request)[:2])
    return result


@router.post('/quota/validate',dependencies=[Depends(private)])
async def validate(payload: Credentials, request: Request):
    store=AccessStore();ip,visitor,*_=identities(request)
    await call(store.throttle,['key:'+ip],30,300)
    from backend.web_v2.quota_feedback import bind
    from backend.web_v2.compute_access import ComputeStore
    await call(bind,store,visitor,payload.email,payload.key)
    result = await call(store.status,ip,visitor,payload.email,payload.key)
    result['compute'] = await call(ComputeStore().status,actor=(ip,visitor))
    return result


@router.post('/admin/login',dependencies=[Depends(private)])
async def login(payload: Login, request: Request, response: Response):
    token, user=await call(AccessStore().login,payload.email,payload.password,fingerprint(client_ip(request)))
    response.set_cookie(admin_cookie(),token,max_age=8*3600,httponly=True,samesite='strict',secure=secure_request(request),path='/api/admin')
    return user


@router.get('/admin/me')
async def me(user=Depends(admin)): return user


@router.post('/admin/logout')
async def logout(request: Request, response: Response, user=Depends(admin)):
    with AccessStore().connect(True) as c: c.execute('DELETE FROM admin_sessions WHERE token_hash=?',(fingerprint(request.cookies.get(admin_cookie(),'') or request.cookies.get(ADMIN_COOKIE,'')),))
    response.delete_cookie(admin_cookie(),path='/api/admin')
    response.delete_cookie(ADMIN_COOKIE,path='/api/admin')
    return {'ok':True}


@router.get('/admin/admins')
async def admins(user=Depends(admin)): return {'items':await call(AccessStore().admins)}


@router.post('/admin/admins')
async def add_admin(payload: Administrator,user=Depends(admin)):
    return await call(AccessStore().save_admin,user,payload.model_dump())


@router.put('/admin/admins/{ident}')
async def edit_admin(ident: str,payload: Administrator,user=Depends(admin)):
    return await call(AccessStore().save_admin,user,payload.model_dump(),ident)


@router.delete('/admin/admins/{ident}')
async def delete_admin(ident: str,user=Depends(admin)):
    await call(AccessStore().delete_admin,user,ident);return {'ok':True}


@router.get('/admin/grants')
async def grants(q: str='',user=Depends(admin)):
    return {'items':await call(AccessStore().grants,q[:200])}


@router.post('/admin/grants')
async def add_grant(payload: Grant,user=Depends(admin)):
    return await call(AccessStore().save_grant,user,payload.model_dump())


@router.put('/admin/grants/{ident}')
async def edit_grant(ident: str,payload: Grant,user=Depends(admin)):
    return await call(AccessStore().save_grant,user,payload.model_dump(),ident)


@router.post('/admin/grants/{ident}/rotate')
async def rotate_grant(ident: str,payload: Grant,user=Depends(admin)):
    return await call(AccessStore().save_grant,user,payload.model_dump(),ident,True)


@router.delete('/admin/grants/{ident}')
async def delete_grant(ident: str,user=Depends(admin)):
    await call(AccessStore().delete_grant,user,ident);return {'ok':True}


@router.get('/admin/audit')
async def audit(user=Depends(admin)):
    with AccessStore().connect() as c:
        return {'items':[dict(r) for r in c.execute('SELECT * FROM audit ORDER BY id DESC LIMIT 100')]}


def usage_filters(days: int=Query(7,ge=1,le=90), mode: Literal['','agent','scientist','researcher','tools']='', session_id: str=Query('',max_length=128)):
    return dict(days=days,mode=mode,session_id=session_id)


@router.get('/admin/usage')
async def usage_overview(filters=Depends(usage_filters), user=Depends(admin)):
    from backend.web_v2.usage_store import UsageStore
    return await call(lambda: UsageStore().overview(**filters))


@router.get('/admin/usage/jobs')
async def usage_jobs(filters=Depends(usage_filters), offset: int=Query(0,ge=0,le=1000000), limit: int=Query(25,ge=1,le=100), user=Depends(admin)):
    from backend.web_v2.usage_store import UsageStore
    return await call(lambda: UsageStore().jobs(**filters,offset=offset,limit=limit))


@router.get('/admin/usage/events')
async def usage_events(filters=Depends(usage_filters), kind: Literal['','model','tool']='', offset: int=Query(0,ge=0,le=1000000), limit: int=Query(25,ge=1,le=100), user=Depends(admin)):
    from backend.web_v2.usage_store import UsageStore
    return await call(lambda: UsageStore().events(**filters,kind=kind,offset=offset,limit=limit))


@router.get('/admin/usage/sessions')
async def usage_sessions(filters=Depends(usage_filters), offset: int=Query(0,ge=0,le=1000000), limit: int=Query(25,ge=1,le=100), user=Depends(admin)):
    from backend.web_v2.usage_store import UsageStore
    return await call(lambda: UsageStore().sessions(**filters,offset=offset,limit=limit))


class CapabilityChange(BaseModel):
    model_config=ConfigDict(extra='forbid')
    key: str=Field(min_length=1,max_length=300)
    enabled: StrictBool
    revision: StrictInt=Field(ge=0)


@router.get('/admin/capabilities')
async def capabilities(user=Depends(admin)):
    from backend.web_v2.capability_admin import inventory, public_inventory
    return public_inventory(await inventory())


@router.put('/admin/capabilities')
async def change_capability(payload: CapabilityChange,user=Depends(admin)):
    from backend.web_v2.capability_admin import inventory
    from mcp_tools.tools.capability_policy import CapabilityStore
    data=await inventory()
    row=next((r for r in data['items'] if r['key']==payload.key),None)
    if not row:raise HTTPException(404,'Tool or model not found')
    try:
        revision=await run_in_threadpool(CapabilityStore().set,payload.key,payload.enabled,payload.revision,row['selector'],user['email'])
    except ValueError as exc:raise HTTPException(409,str(exc)) from None
    store=AccessStore()
    with store.connect(True) as c: store.log(c,user['email'],'capability.enable' if payload.enabled else 'capability.disable',payload.key)
    return {'ok':True,'revision':revision,'mode':data['mode']}


def secret_env_key(key):
    return any(part in key.upper() for part in ('KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL'))


@router.get('/admin/env')
async def admin_env(user=Depends(admin)):
    from backend.web_v2.settings_api import read_env_entries
    data=await run_in_threadpool(read_env_entries)
    data['entries']=[{**e,'value':'••••••••' if secret_env_key(e['key']) and e['value'] else e['value']} for e in data['entries']]
    return data


from backend.web_v2.settings_api import UpdateEnvRequest


@router.post('/admin/env')
async def admin_save_env(payload: UpdateEnvRequest,user=Depends(admin)):
    from backend.web_v2.settings_api import write_env_entries
    cleaned=payload.model_copy(update={'entries':[e for e in payload.entries if not (secret_env_key(e.key) and e.value=='••••••••')]})
    result=await run_in_threadpool(write_env_entries,cleaned)
    # Record keys only: never credentials or their previous values.
    store=AccessStore()
    with store.connect(True) as c:
        store.log(c,user['email'],'settings.update',','.join(e.key for e in cleaned.entries))
    return result


@router.post('/quota/disconnect',dependencies=[Depends(private)])
async def disconnect_grant(request: Request):
    from backend.web_v2.quota_feedback import unbind
    await call(unbind,AccessStore(),identities(request)[1])
    return {'ok':True}


@router.get('/admin/allowances')
async def allowance_overview(user=Depends(admin)):
    from backend.web_v2.quota_feedback import overview
    from backend.web_v2.compute_access import ComputeStore
    store=ComputeStore()
    return {'overview':await call(overview,store)}


class ResetAllowances(BaseModel):
    model_config=ConfigDict(extra='forbid')
    confirm: Literal['RESET']


@router.post('/admin/allowances/reset')
async def reset_allowances(payload: ResetAllowances,user=Depends(admin)):
    from backend.web_v2.quota_feedback import reset
    return await call(reset,AccessStore(),user)
