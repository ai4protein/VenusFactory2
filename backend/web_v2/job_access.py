"""Shared browser identity and private-file checks for job HTTP adapters."""
from contextvars import ContextVar
import hashlib
import re
import secrets
from fastapi import HTTPException, Request, Response
from backend.web_v2.job_records import JobRecords
from backend.web_v2.http_security import require_same_origin, secure_request

COOKIES = {'platform': 'vf_model_workspace', 'custom': 'vf_custom_jobs', 'report': 'vf_research_jobs', 'mcp': 'vf_tool_jobs'}
request_context = ContextVar('job_request', default=None)


def browser_owner(request, response, source, session=None):
    require_same_origin(request)
    cache = getattr(request.state, 'job_owners', {})
    if source in cache:
        return cache[source]
    cookie = COOKIES[source]
    old = request.cookies.get(cookie, '')
    if session:
        token, owner = session(old)
    else:
        token = old if re.fullmatch('[a-f0-9]{64}', old) else secrets.token_hex(32)
        owner = hashlib.sha256(token.encode()).hexdigest()
    # Preserve existing identities while making them available to result routes.
    response.set_cookie(cookie, token, httponly=True, secure=secure_request(request), samesite='strict', max_age=365*86400, path='/api')
    response.delete_cookie(cookie, path='/api/model-platform' if source == 'platform' else '/api/custom-model')
    response.headers['Cache-Control'] = 'private, no-store'
    response.headers['Vary'] = 'Cookie'
    cache[source] = owner
    request.state.job_owners = cache
    return owner


def request_owners(request):
    owners = dict(getattr(request.state, 'job_owners', {}))
    for source, cookie in COOKIES.items():
        token = request.cookies.get(cookie, '')
        if source not in owners and re.fullmatch('[a-f0-9]{64}', token):
            owners[source] = hashlib.sha256(token.encode()).hexdigest()
    return owners


def guard_artifact(path, request=None):
    request = request or request_context.get()
    if request is None:
        return  # Internal computation helpers are not HTTP reads.
    try:
        from backend.web_v2.online_visibility import guard_generic_file
        guard_generic_file(path)
        JobRecords().check_artifact(path, request_owners(request))
    except FileNotFoundError:
        raise HTTPException(404, 'File not found') from None


async def custom_request_scope(request: Request, response: Response):
    require_same_origin(request)
    token = request_context.set(request)
    try:
        yield
    finally:
        request_context.reset(token)
