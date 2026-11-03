"""Shared browser/proxy security boundary for API identities and mutations."""
import ipaddress
import os
from urllib.parse import urlsplit

from fastapi import HTTPException


def normalized_origin(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.path not in {'', '/'} or parsed.query or parsed.fragment):
        raise ValueError('Invalid origin')
    port = parsed.port if parsed.port is not None else (443 if parsed.scheme == 'https' else 80)
    return parsed.scheme, parsed.hostname, port


def client_ip(request):
    # A proxy must overwrite/append XFF and be explicitly trusted. Never trust
    # a client-supplied leftmost address. Uvicorn must use --no-proxy-headers.
    peer = request.client.host if request.client else 'unknown'
    networks = []
    for value in os.getenv('VENUS_TRUSTED_PROXIES', '').split(','):
        try: networks.append(ipaddress.ip_network(value.strip()))
        except ValueError: pass
    def trusted(value):
        try: return any(ipaddress.ip_address(value) in n for n in networks)
        except ValueError: return False
    if trusted(peer):
        chain = [x.strip() for x in request.headers.get('x-forwarded-for', '').split(',') if x.strip()] + [peer]
        for value in reversed(chain):
            try: ipaddress.ip_address(value)
            except ValueError: return peer
            if not trusted(value):
                peer = value
                break
    try:
        address = ipaddress.ip_address(peer)
        # Group IPv6 privacy addresses by /64.
        return str(ipaddress.ip_network(str(address) + '/64', strict=False)) if address.version == 6 else str(address)
    except ValueError: return peer


def secure_request(request):
    if request.url.scheme == 'https': return True
    # A TLS-terminating tunnel may not supply forwarded headers. An explicit
    # administrator-configured public origin describes that one host only.
    public_origin = os.getenv('WEBUI_V2_PUBLIC_ORIGIN', '').strip()
    if public_origin:
        try:
            configured = normalized_origin(public_origin)
            if configured[0] == 'https' and configured == normalized_origin('https://' + request.headers.get('host', '')):
                return True
        except ValueError:
            pass
    peer = request.client.host if request.client else ''
    if request.headers.get('x-forwarded-proto', '').strip().lower() != 'https': return False
    for value in os.getenv('VENUS_TRUSTED_PROXIES', '').split(','):
        try:
            if ipaddress.ip_address(peer) in ipaddress.ip_network(value.strip()): return True
        except ValueError: pass
    return False


def require_same_origin(request):
    if request.method in {'GET', 'HEAD', 'OPTIONS'}:
        return
    rejected = request.headers.get('sec-fetch-site') == 'cross-site'
    origin = request.headers.get('origin')
    if origin:
        try:
            scheme = 'https' if secure_request(request) else 'http'
            expected = normalized_origin(scheme+'://'+request.headers.get('host', ''))
            rejected = rejected or normalized_origin(origin) != expected
        except ValueError:
            rejected = True
    if rejected:
        raise HTTPException(403, 'Use this workspace to submit changes')
