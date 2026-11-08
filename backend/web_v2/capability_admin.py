"""Authenticated catalogue controls; imports are lazy to keep sign-in lightweight."""
import asyncio
from mcp_tools.tools.capability_policy import CapabilityStore, canonical, selector, mode


async def inventory():
    from mcp_tools.tools.tools_agent_hub import ALL_TOOLS
    from mcp_tools.scheduler.catalog import classify
    from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
    from backend.web_v2.model_platform import Platform, read
    from mcp_tools.mcp_server import mcp
    tools = {canonical(t.name): t for t in ALL_TOOLS}
    for tool in await mcp.list_tools(): tools.setdefault(canonical(tool.name), tool)
    rows = []
    for name, tool in tools.items():
        info = classify(name, getattr(tool,'metadata',None))
        rows.append({'key':'tool:'+name,'name':name,'kind':'tool','category':info['category'],
                     'description':tool.description or '', 'available':is_tool_allowed(name),
                     'selector':{},'category_label':info['category_label'],'category_label_zh':info['category_label_zh']})
    from agent.harness.research_workspace import RESEARCH_TOOLS
    for name in RESEARCH_TOOLS:
        rows.append({'key':'tool:'+name,'name':name,'kind':'tool','category':'workspace', 'description':name.replace('_',' '),
                     'available':is_tool_allowed(name),'selector':{},'category_label':'Scientist workspace','category_label_zh':'Scientist 研究工作区'})
    for name in ['list_session_files','read_session_file','write_session_file','run_session_python','run_session_shell']:
        info=classify(name)
        rows.append({'key':'tool:'+name,'name':name,'kind':'tool','category':info['category'],'description':name.replace('_',' '),
                     'available':is_tool_allowed(name),'selector':{},'category_label':info['category_label'],'category_label_zh':info['category_label_zh']})
    platform = Platform()
    cards, issues = await asyncio.to_thread(platform.shared, True)
    for path in (platform.root/'models').glob('*.json'):
        item=read(path)
        cards.append(platform.model_card(item, ''))
    for card in cards:
        available=platform.availability(card)
        rows.append({'key':'model:'+card['id'],'name':card['name'],'kind':'model','category':card['category'],
                     'description':card.get('description',''),'available':available['available'],
                     'reason':available.get('disabled_reason'),'selector':selector(card),
                     **{k:available.get(k) for k in ('model_type','runtime_id','backbone','training_dataset','provenance','version')}})
    saved=CapabilityStore().rows()
    for row in rows:
        setting=saved.get(row['key'],{})
        row.update(enabled=bool(setting.get('enabled',True)), revision=setting.get('revision',0))
    return {'mode':mode(),'items':rows,'unavailable_sources':issues}


def public_inventory(data):
    return {**data,'items':[{k:v for k,v in row.items() if k!='selector'} for row in data['items']]}
