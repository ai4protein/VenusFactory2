"""Repair only exact, unambiguous names from the advertised MCP schema."""
import json


def normalize(raw, streaming, tools):
    names={t['function']['name'] for t in tools}
    aliases={}
    for name in names:
        if name.startswith('mcp__venusfactory__'):
            bare=name[len('mcp__venusfactory__'):]
            if bare not in names: aliases[bare]=name
    repaired=[]
    def canonical(name):
        if name in names:return name
        if name in aliases:
            repaired.append({'from':name,'to':aliases[name]});return aliases[name]
        raise ValueError('Model selected a tool outside the advertised session interface')
    if not streaming:
        obj=json.loads(raw)
        for choice in obj.get('choices',[]):
            for call in choice.get('message',{}).get('tool_calls',[]):
                fn=call['function'];fn['name']=canonical(fn['name'])
        return json.dumps(obj,ensure_ascii=False).encode(),repaired
    lines=raw.splitlines(keepends=True);groups={};objects={}
    for i,line in enumerate(lines):
        if not line.startswith(b'data:'):continue
        data=line[5:].strip()
        if not data or data==b'[DONE]':continue
        obj=json.loads(data);objects[i]=obj
        for choice in obj.get('choices',[]):
            for call in choice.get('delta',{}).get('tool_calls',[]):
                fn=call.get('function',{})
                if 'name' in fn:groups.setdefault((choice.get('index',0),call.get('index',0)),[]).append(fn)
    for pieces in groups.values():
        full=canonical(''.join(p['name'] for p in pieces))
        pieces[0]['name']=full
        for p in pieces[1:]:p.pop('name',None)
    for i,obj in objects.items():lines[i]=b'data: '+json.dumps(obj,ensure_ascii=False).encode()+b'\n'
    return b''.join(lines),repaired
