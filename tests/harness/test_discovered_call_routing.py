import json
from types import SimpleNamespace

import pytest

from agent.harness.tool_disclosure import ToolDisclosure, DisclosureError
from agent.harness.tool_response_names import normalize, PREFIX

TOOLS = [{'function': {'name': PREFIX + name}} for name in
         ('search_tools', 'describe_tool', 'call_tool')]


def response(name, arguments='{"query":"training"}'):
    return json.dumps({'choices': [{'message': {'tool_calls': [
        {'id': 'call_1', 'type': 'function', 'function': {'name': name, 'arguments': arguments}}
    ]}, 'finish_reason': 'tool_calls'}]}).encode()


def test_direct_discovered_call_uses_broker_and_keeps_permission_checks():
    raw, repaired = normalize(response(PREFIX + 'query_platform_manual'), False, TOOLS)
    call = json.loads(raw)['choices'][0]['message']['tool_calls'][0]
    assert call['id'] == 'call_1'
    assert call['function']['name'] == PREFIX + 'call_tool'
    args = json.loads(call['function']['arguments'])
    assert args == {'name': 'query_platform_manual', 'arguments': {'query': 'training'}}
    assert repaired[0]['via'] == 'inspected_broker'
    enabled = True
    tool = SimpleNamespace(name='query_platform_manual', description='Read the platform manual.')
    d = ToolDisclosure({tool.name: tool}, schema=lambda t: {}, allowed=lambda t: enabled)
    with pytest.raises(DisclosureError, match='describe_tool'):
        d.resolve('call_tool', args)
    d.metadata('describe_tool', {'name': tool.name})
    assert d.resolve('call_tool', args)[2] == {'query': 'training'}
    enabled = False
    with pytest.raises(DisclosureError, match='unavailable'):
        d.resolve('call_tool', args)


def test_streamed_name_and_argument_fragments_are_wrapped_once():
    chunks = [
        {'index': 0, 'id': 'call_1', 'function': {'name': PREFIX + 'query_', 'arguments': '{"query":'}},
        {'index': 0, 'function': {'name': 'platform_manual', 'arguments': '"training"}'}},
    ]
    raw = b''.join(b'data: ' + json.dumps({'choices': [{'index': 0, 'delta': {'tool_calls': [c]}}]}).encode() + b'\n\n' for c in chunks) + b'data: [DONE]\n\n'
    output, repaired = normalize(raw, True, TOOLS)
    calls = [json.loads(line[5:])['choices'][0]['delta']['tool_calls'][0]
             for line in output.splitlines() if line.startswith(b'data: {')]
    assert calls[0]['id'] == 'call_1' and len(repaired) == 1
    assert ''.join(c['function'].get('name', '') for c in calls) == PREFIX + 'call_tool'
    args = json.loads(''.join(c['function'].get('arguments', '') for c in calls))
    assert args == {'name': 'query_platform_manual', 'arguments': {'query': 'training'}}


@pytest.mark.parametrize('name', ['bash', 'mcp__other__query', PREFIX + '../secret'])
def test_nonplatform_tools_are_not_repaired(name):
    with pytest.raises(ValueError):
        normalize(response(name), False, TOOLS)


def test_broker_must_be_advertised_and_arguments_must_be_an_object():
    with pytest.raises(ValueError):
        normalize(response(PREFIX + 'query_platform_manual'), False, TOOLS[:2])
    with pytest.raises(ValueError):
        normalize(response(PREFIX + 'query_platform_manual', '[]'), False, TOOLS)
