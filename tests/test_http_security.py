import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.testclient import TestClient

from backend.web_v2.http_security import require_same_origin, secure_request
from backend.web_v2.job_access import browser_owner
from backend.web_v2.online_access import VisitorMiddleware


def request(origin=None, *, host='venus.example', proto=None, peer='127.0.0.1', site=None, scheme='http'):
    headers=[(b'host',host.encode())]
    for key,value in [('origin',origin),('x-forwarded-proto',proto),('sec-fetch-site',site)]:
        if value is not None:headers.append((key.encode(),value.encode()))
    return Request({'type':'http','method':'POST','scheme':scheme,'path':'/api/jobs','headers':headers,'client':(peer,1234),'server':('venus.example',80),'query_string':b''})


@pytest.mark.parametrize('origin', ['https://evil.example','http://venus.example:8080','http://venus.example:0','null','https://venus.example','http://venus.example/path','http://user@venus.example','http://venus.example:bad','http://[invalid'])
def test_origin_rejects_cross_site_scheme_port_and_malformed_input(origin, monkeypatch):
    monkeypatch.delenv('VENUS_TRUSTED_PROXIES',raising=False)
    with pytest.raises(HTTPException) as error:require_same_origin(request(origin))
    assert error.value.status_code==403


def test_normalized_origin_and_command_line_clients_remain_supported(monkeypatch):
    monkeypatch.delenv('VENUS_TRUSTED_PROXIES',raising=False)
    require_same_origin(request('http://VENUS.EXAMPLE:80'))
    require_same_origin(request())
    with pytest.raises(HTTPException):require_same_origin(request(site='cross-site'))


def test_only_explicitly_trusted_proxies_can_select_https(monkeypatch):
    monkeypatch.setenv('VENUS_TRUSTED_PROXIES','127.0.0.1/32')
    trusted=request('https://venus.example',proto='https')
    assert secure_request(trusted);require_same_origin(trusted)
    forged=request('https://venus.example',proto='https',peer='198.51.100.42')
    assert not secure_request(forged)
    with pytest.raises(HTTPException):require_same_origin(forged)


@pytest.mark.parametrize('source',['platform','custom','report','mcp'])
def test_private_job_cookies_are_secure_behind_tls_proxy(source,monkeypatch):
    monkeypatch.setenv('VENUS_TRUSTED_PROXIES','127.0.0.1/32')
    response=Response();browser_owner(request('https://venus.example',proto='https'),response,source)
    cookie=next(h for h in response.headers.getlist('set-cookie')if 'Max-Age=31536000' in h)
    assert all(flag in cookie for flag in ['Secure','HttpOnly','SameSite=strict','Path=/api'])


def test_api_middleware_protects_legacy_mutations_before_the_handler(monkeypatch):
    monkeypatch.delenv('VENUS_TRUSTED_PROXIES',raising=False)
    called=[];app=FastAPI();app.add_middleware(VisitorMiddleware)
    @app.post('/api/legacy/run')
    def mutation():called.append(True);return {'ok':True}
    with TestClient(app)as client:
        assert client.post('/api/legacy/run',headers={'Origin':'https://evil.example'}).status_code==403
        assert not called
        assert client.post('/api/legacy/run',headers={'Origin':'http://testserver'}).status_code==200
        assert len(called)==1
