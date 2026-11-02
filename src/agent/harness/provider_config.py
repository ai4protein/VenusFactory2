"""Opt-in, server-owned official API configuration for local deployments.

No model is enabled by a key alone. Exact model IDs are supplied by the operator;
online deployments continue to use the independently maintained free catalog.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from .contracts import ModelBinding


@dataclass(frozen=True)
class Provider:
    id: str
    prefix: str
    base_url: str
    aliases: tuple[str, ...] = ()


PROVIDERS = (
    Provider('openai', 'OPENAI', 'https://api.openai.com/v1'),
    Provider('anthropic', 'ANTHROPIC', 'https://api.anthropic.com/v1'),
    Provider('xai', 'XAI', 'https://api.x.ai/v1', ('GROK_API_KEY',)),
    Provider('glm', 'GLM', 'https://open.bigmodel.cn/api/paas/v4', ('ZHIPU_API_KEY', 'ZAI_API_KEY')),
    Provider('minimax', 'MINIMAX', 'https://api.minimax.io/v1'),
    Provider('deepseek', 'DEEPSEEK', 'https://api.deepseek.com/v1'),
    Provider('google', 'GEMINI', 'https://generativelanguage.googleapis.com/v1beta/openai', ('GOOGLE_API_KEY',)),
    Provider('alibaba', 'QWEN', 'https://dashscope.aliyuncs.com/compatible-mode/v1', ('DASHSCOPE_API_KEY',)),
    Provider('moonshot', 'MOONSHOT', 'https://api.moonshot.cn/v1'),
)


def online():
    return os.getenv('WEBUI_V2_MODE', 'local').strip().lower() == 'online'


def configuration_keys():
    return tuple(key for p in PROVIDERS for key in (
        *(p.prefix + suffix for suffix in ('_API_KEY', '_BASE_URL', '_MODEL', '_MODELS', '_CONTEXT_TOKENS', '_MAX_OUTPUT_TOKENS')),
        *p.aliases,
    )) + ('ANTHROPIC_WORKSPACE_ID',)


def _positive(values, name, default, maximum):
    raw = values.get(name, '').strip()
    value = int(raw) if raw else default
    if value < 1 or value > maximum:
        raise ValueError(name + ' is outside the supported range')
    return value


def configured_models(values=None, *, require_key=False):
    """Public configuration only; credentials are represented by env names."""
    values = os.environ if values is None else values
    rows = []
    for p in PROVIDERS:
        names = ','.join(values.get(p.prefix + suffix, '') for suffix in ('_MODEL', '_MODELS'))
        models = list(dict.fromkeys(n.strip() for n in names.split(',') if n.strip()))
        if not models:
            continue
        if len(models) > 32 or any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./:@+-]{0,159}', m) for m in models):
            raise ValueError(p.prefix + '_MODELS must contain at most 32 exact model IDs')
        key_names = (p.prefix + '_API_KEY', *p.aliases)
        key_env = next((k for k in key_names if values.get(k, '').strip()), key_names[0])
        if require_key and not values.get(key_env, '').strip():
            continue
        base = (values.get(p.prefix + '_BASE_URL', '').strip() or p.base_url).rstrip('/')
        url = urlsplit(base)
        if url.scheme != 'https' or not url.hostname or url.port not in (None, 443) or url.username or url.password or url.query or url.fragment:
            raise ValueError(p.prefix + '_BASE_URL must be an HTTPS API base without credentials or query parameters')
        context = _positive(values, p.prefix + '_CONTEXT_TOKENS', 131072, 10000000)
        ceiling = _positive(values, p.prefix + '_MAX_OUTPUT_TOKENS', 4096, 1000000)
        if ceiling >= context:
            raise ValueError(p.prefix + '_MAX_OUTPUT_TOKENS must be smaller than context capacity')
        for model in models:
            # Identity does not change when comma-separated model order changes.
            mid = 'env-' + p.id + '-' + hashlib.sha256(model.encode()).hexdigest()[:16]
            rows.append(dict(id=mid, label=model, binding=ModelBinding(
                provider=p.id, model=model, base_url=base, credential_env=key_env, context_tokens=context,
            ), max_output_tokens=ceiling))
    return rows


def local_route(model, base_url, protocol):
    if online():
        return None
    # OpenRouter remains governed by its explicit zero-price catalog.
    if urlsplit(base_url).hostname == 'openrouter.ai':
        return None
    for row in configured_models(require_key=True):
        b = row['binding']
        protocols = {'chat_completions', 'responses'} if b.provider == 'openai' else {'chat_completions'}
        if (model, base_url.rstrip('/')) == (b.model, b.base_url) and protocol in protocols:
            return row
    return None


def authorize_local(model, base_url, protocol, output_tokens):
    row = local_route(model, base_url, protocol)
    if row is None:
        return None
    if type(output_tokens) is not int or not 0 < output_tokens <= row['max_output_tokens']:
        raise ValueError('Model output exceeds the configured provider limit')
    public = dict(model=model, base_url=base_url, protocol=protocol, max_output_tokens=row['max_output_tokens'])
    return dict(policy_id='local-configured-provider.v1', scope='local_configured_model',
                policy_sha256=hashlib.sha256(json.dumps(public, sort_keys=True).encode()).hexdigest())


def adapt_request(payload, base_url, protocol):
    """Only explicitly configured local routes receive vendor parameter mapping."""
    row = local_route(payload['model'], base_url, protocol)
    if row is None:
        return payload
    result = dict(payload)
    # Provider defaults are deliberate: shared free-route thinking/temperature
    # controls are not portable to official reasoning models.
    for key in ('thinking', 'temperature', 'top_p', 'reasoning_effort'):
        result.pop(key, None)
    if row['binding'].provider == 'openai' and protocol == 'chat_completions':
        result['max_completion_tokens'] = result.pop('max_tokens')
    return result


def request_headers(base_url, api_key):
    headers = {'Authorization': 'Bearer ' + api_key}
    # Workspace identifiers must never be sent to another provider/proxy.
    if urlsplit(base_url).hostname == 'api.anthropic.com' and os.getenv('ANTHROPIC_WORKSPACE_ID', '').strip():
        headers['anthropic-workspace-id'] = os.environ['ANTHROPIC_WORKSPACE_ID'].strip()
    return headers

def matches_snapshot(requested, returned, base_url=None):
    """Accept dated snapshots of configured official aliases, not other models."""
    if not isinstance(returned, str) or online():
        return False
    for row in configured_models(require_key=True):
        b = row['binding']
        if b.model != requested or (base_url is not None and b.base_url != base_url.rstrip('/')):
            continue
        if b.provider == 'openai':
            return re.fullmatch(re.escape(requested) + r'-[0-9]{4}-[0-9]{2}-[0-9]{2}', returned) is not None
        if b.provider == 'anthropic' and requested.endswith('-latest'):
            return re.fullmatch(re.escape(requested[:-6]) + r'[0-9]{8}', returned) is not None
    return False
