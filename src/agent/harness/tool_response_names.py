"""Normalize MCP aliases and route discovered names through the inspected broker."""
import json

PREFIX = 'mcp__venusfactory__'


def normalize(raw, streaming, tools):
    names = {t['function']['name'] for t in tools}
    aliases = {name[len(PREFIX):]: name for name in names
               if name.startswith(PREFIX) and name[len(PREFIX):] not in names}
    broker = PREFIX + 'call_tool'
    discovery = {PREFIX + name for name in ('search_tools', 'describe_tool', 'call_tool')} <= names
    repaired = []

    def canonical(name):
        if name in names:
            return name, None
        if name in aliases:
            repaired.append({'from': name, 'to': aliases[name]})
            return aliases[name], None
        if discovery and name.startswith(PREFIX):
            target = name[len(PREFIX):]
            if target and len(target) <= 128 and all(c.isascii() and (c.isalnum() or c == '_') for c in target):
                # The broker still requires prior inspection, current permission and valid inputs.
                # This does not add the discovered tool to the native client's executable surface.
                repaired.append({'from': name, 'to': broker, 'via': 'inspected_broker'})
                return broker, target
        raise ValueError('Model selected a tool outside the advertised session interface')

    def envelope(target, arguments):
        value = json.loads(arguments or '{}')
        if not isinstance(value, dict):
            raise ValueError('Discovered tool arguments must be an object')
        return json.dumps({'name': target, 'arguments': value}, ensure_ascii=False)

    if not streaming:
        obj = json.loads(raw)
        for choice in obj.get('choices', []):
            for call in choice.get('message', {}).get('tool_calls', []):
                fn = call['function']
                fn['name'], target = canonical(fn['name'])
                if target is not None:
                    fn['arguments'] = envelope(target, fn.get('arguments'))
        return json.dumps(obj, ensure_ascii=False).encode(), repaired

    lines, groups, objects = raw.splitlines(keepends=True), {}, {}
    for i, line in enumerate(lines):
        if not line.startswith(b'data:'):
            continue
        data = line[5:].strip()
        if not data or data == b'[DONE]':
            continue
        obj = objects[i] = json.loads(data)
        for choice in obj.get('choices', []):
            for call in choice.get('delta', {}).get('tool_calls', []):
                fn = call.setdefault('function', {})
                groups.setdefault((choice.get('index', 0), call.get('index', 0)), []).append(fn)
    for pieces in groups.values():
        full, target = canonical(''.join(p.get('name', '') for p in pieces))
        arguments = envelope(target, ''.join(p.get('arguments', '') for p in pieces)) if target else None
        pieces[0]['name'] = full
        if target:
            pieces[0]['arguments'] = arguments
        for piece in pieces[1:]:
            piece.pop('name', None)
            if target:
                piece.pop('arguments', None)
    for i, obj in objects.items():
        lines[i] = b'data: ' + json.dumps(obj, ensure_ascii=False).encode() + b'\n'
    return b''.join(lines), repaired
