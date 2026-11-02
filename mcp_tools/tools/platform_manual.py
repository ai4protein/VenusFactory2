"""Read-only, allowlisted retrieval over the platform's canonical user tutorials."""
from pathlib import Path
import hashlib
import json
import re
from typing import Literal

ROOT = Path(__file__).resolve().parents[2] / 'docs/manual'
TOOL_DESCRIPTION = (
    'Search or read the current VenusFactory2 Manual/Wiki for platform usage, Chat modes, '
    'training, prediction, workflows, files, quotas and settings. Call before answering platform how-to questions. '
    'Use a question in English/Chinese, or section from the returned catalog to read a topic. '
    'Returns tutorial excerpts with citation_markdown and revision hashes. Copy citation_markdown exactly; '
    'links are app-relative, so never add a domain. No network or private files are read. '
    'For live model/tool availability use search_research_tools instead.'
)
STOPWORDS = {'a','an','the','and','or','to','of','is','in','it','my','how','do','i','can','with','for','what','does','use','please','venusfactory2','venusfactory'}


def _read(relative):
    path = ROOT / relative
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()) or path.is_symlink():
        raise ValueError('Invalid documentation path')
    return resolved.read_text(encoding='utf-8')


def _tokens(text):
    text = text.lower()
    english = {s for s in re.findall(r'[a-z0-9][a-z0-9_-]*',text) if s not in STOPWORDS}
    for span in re.findall(r'[\u3400-\u9fff]+',text):
        english.update(span[i:i+2] for i in range(len(span)-1))
    return english


def _chunks(text):
    lines = text.splitlines()
    headings = [(i, line.lstrip('# ').strip()) for i,line in enumerate(lines) if re.match(r'^#{1,3} ',line)]
    if not headings:
        return [('header-0','',text)]
    return [('header-'+str(n),title,'\n'.join(lines[start:headings[n+1][0] if n+1<len(headings) else len(lines)]).strip())
            for n,(start,title) in enumerate(headings)]


def query_manual(query='', section='', language='auto', limit=5):
    from mcp_tools.tools.runtime_tool_policy import assert_tool_allowed, is_online_mode
    assert_tool_allowed('query_platform_manual')
    if language not in {'auto','en','zh'}:raise ValueError('language must be auto, en or zh')
    if len(query)>1000:raise ValueError('Keep the documentation question within 1000 characters')
    if type(limit) is not int or not 1<=limit<=8:raise ValueError('limit must be between 1 and 8')
    lang = ('zh' if re.search(r'[\u3400-\u9fff]',query) else 'en') if language=='auto' else language
    catalog=json.loads(_read('catalog.json'))
    entries=catalog['sections'];by_key={s['key']:s for s in entries}
    alias={a:s['key'] for s in entries for a in s['aliases']}
    section=alias.get(section,section)
    if section and section not in by_key:
        raise ValueError('Unknown manual section. Use a section from the documentation catalog.')
    overview=[{'section':s['key'],'title':s['title'][lang],'summary':s['summary'][lang],'url':f"/{lang}/manual/{s['key']}"} for s in entries]
    result={'success':True,'language':lang,'updated':catalog['updated'],'mode':'online' if is_online_mode() else 'local',
            'catalog':overview,'results':[]}
    if not query.strip() and not section:return result
    words=_tokens(query);ranked=[]
    for item in entries:
        if section and item['key']!=section:continue
        text=_read(item['files'][lang]);digest=hashlib.sha256(text.encode()).hexdigest()
        section_words=_tokens(item['key']+' '+item['title'][lang]+' '+item['summary'][lang])
        for anchor,title,body in _chunks(text):
            score=3*len(words & _tokens(title))+len(words & _tokens(body))+2*len(words & section_words)
            if query.strip() and score==0:continue
            url=f"/{lang}/manual/{item['key']}#{anchor}"
            ranked.append((score,{'section':item['key'],'title':item['title'][lang], 'heading':title,
                'url':url, 'citation_markdown':f"[{item['title'][lang]} — {title}]({url})",
                'content':body,'content_sha256':digest}))
    if query.strip():ranked.sort(key=lambda row:row[0],reverse=True)
    budget=14000
    for _,row in ranked[:limit]:
        body=row['content'];row['content']=body[:min(budget,4000)]
        row['truncated']=len(row['content'])<len(body)
        budget-=len(row['content'])
        result['results'].append(row)
        if budget<=0:break
    result['matched_sections']=len({row['section'] for _,row in ranked})
    result['guidance']='Copy citation_markdown exactly, including its app-relative URL. Never guess or prepend a domain: the user may be on a local or online deployment. State deployment-specific availability separately; do not infer enabled models or private session state from a tutorial.'
    return result


def register_manual_tool(server):
    @server.tool(name='query_platform_manual', description=TOOL_DESCRIPTION)
    def query_platform_manual(query: str='',section: str='',language: Literal['auto','en','zh']='auto',limit: int=5) -> dict:
        return query_manual(query,section,language,limit)
