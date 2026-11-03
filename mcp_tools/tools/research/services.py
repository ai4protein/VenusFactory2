"""Free public protein services with bounded requests and explicit provenance."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urlsplit
import xml.etree.ElementTree as ET

import httpx

CATALOG = [
    ('blast', 'NCBI BLAST', 'NCBI 序列同源检索', 'sequence', 'https://blast.ncbi.nlm.nih.gov/Blast.cgi'),
    ('foldseek_remote', 'Foldseek server', 'Foldseek 结构相似检索', 'structure', 'https://search.foldseek.com/'),
    ('interproscan', 'InterProScan', 'InterProScan 序列功能分析', 'sequence', 'https://www.ebi.ac.uk/jdispatcher/pfa/iprscan5'),
    ('s4pred', 'S4PRED', 'S4PRED 二级结构预测', 'sequence', 'https://github.com/psipred/s4pred'),
    ('interpro', 'InterPro / Pfam', 'InterPro / Pfam 结构域', 'accession', 'https://www.ebi.ac.uk/interpro/'),
    ('quickgo', 'QuickGO', 'QuickGO 功能注释', 'accession', 'https://www.ebi.ac.uk/QuickGO/'),
    ('string', 'STRING', 'STRING 互作网络', 'accession', 'https://string-db.org/'),
    ('kegg', 'KEGG pathways & function', 'KEGG 通路与功能', 'accession', 'https://www.kegg.jp/'),
    ('reactome', 'Reactome', 'Reactome 生物通路', 'accession', 'https://reactome.org/'),
    ('alphafold', 'AlphaFold DB', 'AlphaFold DB 已有结构', 'accession', 'https://alphafold.ebi.ac.uk/'),
    ('pdbe', 'PDBe / PDB', 'PDBe / PDB 实验结构', 'accession', 'https://www.ebi.ac.uk/pdbe/'),
]
SERVICE_IDS = {r[0] for r in CATALOG}
HOSTS = {'blast.ncbi.nlm.nih.gov', 'search.foldseek.com', 'www.ebi.ac.uk', 'reactome.org', 'string-db.org',
         'alphafold.ebi.ac.uk', 'files.rcsb.org', 'rest.uniprot.org', 'eutils.ncbi.nlm.nih.gov',
         'api.platform.opentargets.org', 'rest.ensembl.org', 'pubchem.ncbi.nlm.nih.gov', 'data.rcsb.org', 'rest.kegg.jp'}
ACCESSION = re.compile(r'(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-\d+)?')


class ServiceError(ValueError): pass
class NotApplicable(ServiceError): pass


def options():
    return [dict(id=k, label=en, label_zh=zh, input=needs, url=url, remote=k!='s4pred', free=True) for k,en,zh,needs,url in CATALOG]


def sequence(value):
    value = re.sub(r'\s+', '', value).upper()
    if not 10 <= len(value) <= 5000 or not set(value) <= set('ACDEFGHIKLMNPQRSTVWYBXZJUO'):
        raise ServiceError('Use a protein sequence of 10–5,000 residues')
    return value


def structure_chain(content, chain=''):
    """Return only first-model protein coordinates, never PDB header annotations."""
    from Bio.PDB import PDBParser, PDBIO, Select
    from Bio.SeqUtils import seq1
    if not isinstance(content, str):
        raise ServiceError('Use PDB text')
    from backend.web_v2.upload_limits import validate_upload_size
    try:
        validate_upload_size(len(content.encode('utf-8')))
    except ValueError as exc:
        raise ServiceError(str(exc)) from None
    try:
        model = next(iter(PDBParser(QUIET=True).get_structure('protein', io.StringIO(content))))
        chains = {str(c.id): c for c in model if sum(r.id[0]==' ' and 'CA' in r for r in c) >= 10}
        if chain not in chains:
            if chain or len(chains) != 1: raise ServiceError('Select a protein chain for structure search')
            chain = next(iter(chains))
        chosen = chains[chain]
        seq = sequence(''.join(seq1(r.resname) for r in chosen if r.id[0]==' '))
        class ProteinOnly(Select):
            def accept_chain(self, c): return c.id == chain
            def accept_residue(self, r): return r.id[0] == ' '
        writer=PDBIO();writer.set_structure(model);out=io.StringIO();writer.save(out, ProteinOnly())
        return out.getvalue(), seq, chain
    except ServiceError: raise
    except Exception: raise ServiceError('Could not read protein coordinates') from None


async def rate_gate(host, *, submit=False, rid=''):
    """Cross-worker spacing; NCBI: >=10 s globally and >=60 s for each RID."""
    from backend.web_v2.job_queue import JobQueue
    queue=JobQueue()
    keys={host:10 if host=='blast.ncbi.nlm.nih.gov' else 1}
    if rid: keys[host+':'+rid]=60 if host=='blast.ncbi.nlm.nih.gov' else 5
    while True:
        with queue.connection(write=True) as db:
            db.execute('CREATE TABLE IF NOT EXISTS protein_service_rates (key TEXT PRIMARY KEY, next REAL NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS protein_service_days (key TEXT PRIMARY KEY, count INTEGER NOT NULL)')
            now=time.time()
            wait=max([0]+[(row[0]-now) for key in keys if (row:=db.execute('SELECT next FROM protein_service_rates WHERE key=?',(key,)).fetchone())])
            if wait<=0:
                if submit:
                    day=host+':'+time.strftime('%Y-%m-%d',time.gmtime(now))
                    row=db.execute('SELECT count FROM protein_service_days WHERE key=?',(day,)).fetchone()
                    if row and row[0]>=50: raise ServiceError('Public service daily submission limit reached')
                    db.execute('INSERT INTO protein_service_days VALUES (?,1) ON CONFLICT(key) DO UPDATE SET count=count+1',(day,))
                for key,interval in keys.items():
                    db.execute('INSERT INTO protein_service_rates VALUES (?,?) ON CONFLICT(key) DO UPDATE SET next=excluded.next',(key,now+interval))
                return
        await asyncio.sleep(min(wait,10))


async def request(method, url, *, params=None, data=None, json_body=None, limit=3_000_000, submit=False, rid=''):
    u=urlsplit(url)
    if u.scheme!='https' or u.hostname not in HOSTS or u.port not in {None, 443} or u.username or u.password:
        raise ServiceError('Unsupported public service endpoint')
    from backend.web_v2.job_queue import resource_slot
    async with resource_slot('api:'+u.hostname,kind='public-protein-service'):
        await rate_gate(u.hostname,submit=submit,rid=rid)
        async with httpx.AsyncClient(timeout=30,follow_redirects=False) as client:
            async with client.stream(method,url,params=params,data=data,json=json_body,headers={'Accept':'application/json, text/plain, */*','User-Agent':'VenusFactory2-Researcher/1.0'}) as response:
                if response.status_code in {204,404} and method=='GET':return b''
                if response.status_code!=200:raise ServiceError('Service returned HTTP '+str(response.status_code))
                raw=bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    if len(raw)>limit:raise ServiceError('Public service response exceeds the size limit')
    return bytes(raw)


async def get_json(url, params=None):
    raw=await request('GET',url,params=params)
    return json.loads(raw) if raw else None


def digest_input(service, context):
    relevant={k:context.get(k,'') for k in ('accession','sequence','structure_pdb','structure_chain','pdb_id','source_chain')}
    return hashlib.sha256(json.dumps([service,relevant],sort_keys=True).encode()).hexdigest()


def save(path, value):
    if path is None:return
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False));temp.chmod(0o600);os.replace(temp,path)


def blast_hits(raw):
    # NCBI's legacy XML includes an external DTD declaration. ElementTree does
    # not fetch it; reject entity declarations and keep response size bounded.
    if b'<!ENTITY' in raw.upper():raise ServiceError('Unsupported BLAST XML entities')
    root=ET.fromstring(raw);length=int(root.findtext('.//Iteration_query-len') or root.findtext('BlastOutput_query-len') or 0)
    rows=[]
    for hit in root.findall('.//Hit')[:20]:
        hsp=hit.find('./Hit_hsps/Hsp')
        if hsp is None:continue
        aligned=int(hsp.findtext('Hsp_align-len') or 0);ident=int(hsp.findtext('Hsp_identity') or 0)
        start=int(hsp.findtext('Hsp_query-from') or 0);end=int(hsp.findtext('Hsp_query-to') or 0)
        accession=hit.findtext('Hit_accession') or hit.findtext('Hit_id')
        rows.append({'accession':accession,'title':hit.findtext('Hit_def'),'evalue':float(hsp.findtext('Hsp_evalue') or 0),
                     'bit_score':float(hsp.findtext('Hsp_bit-score') or 0),'identity_pct':round(100*ident/aligned,2) if aligned else None,
                     'query_coverage_pct':round(100*(abs(end-start)+1)/length,2) if length else None,
                     'query_start':start,'query_end':end,'aligned_length':aligned})
    return rows


async def blast(context, row, emit):
    seq=sequence(context.get('sequence','')) if not row.get('job_id') else context.get('sequence','')
    row.update(database='swissprot',program='blastp',evidence_level='sequence_similarity')
    if seq:row['input_length']=len(seq)
    if not row.get('job_id'):
        if row.get('submission_started'):raise ServiceError('Previous submission outcome unknown; no duplicate job submitted')
        row['submission_started']=True;emit()
        args={'CMD':'Put','PROGRAM':'blastp','DATABASE':'swissprot','QUERY':'>query\n'+seq,'HITLIST_SIZE':'20','EXPECT':'0.001','TOOL':'VenusFactory2'}
        if os.getenv('NCBI_EMAIL'):args['EMAIL']=os.environ['NCBI_EMAIL']
        raw=(await request('POST','https://blast.ncbi.nlm.nih.gov/Blast.cgi',data=args,submit=True)).decode()
        match=re.search(r'RID\s*=\s*([A-Z0-9-]+)',raw)
        if not match:raise ServiceError('NCBI did not return a BLAST request ID')
        row.update(job_id=match[1],status='waiting',next_poll=time.time()+60,url='https://blast.ncbi.nlm.nih.gov/Blast.cgi?'+urlencode({'CMD':'Get','RID':match[1]}));emit()
        if context.get('submit_only'):
            row.update(status='pending',summary='Submitted to NCBI; check using job_id after next_poll');return None
    for _ in range(context.get('poll_attempts',6)):
        await asyncio.sleep(max(0,row.get('next_poll',0)-time.time()))
        row['next_poll']=time.time()+60;emit()
        raw=await request('GET','https://blast.ncbi.nlm.nih.gov/Blast.cgi',params={'CMD':'Get','RID':row['job_id'],'FORMAT_TYPE':'XML'},rid=row['job_id'])
        if b'<BlastOutput' in raw:
            hits=blast_hits(raw)
            return {'hits':hits,'count':len(hits),'database':'swissprot','program':'blastp','interpretation':'Sequence similarity supports homology hypotheses; it is not experimental function validation.'}
        text=raw.decode(errors='replace')
        status=re.search(r'Status\s*=\s*(WAITING|READY|FAILED|UNKNOWN)',text)
        if status and status[1] in {'FAILED','UNKNOWN'}:raise ServiceError('NCBI BLAST job failed or expired')
        if not status or status[1]!='WAITING':raise ServiceError('NCBI returned an incomplete or unrecognized BLAST response')
        row.update(status='waiting',summary='NCBI is processing the sequence');emit()
    row.update(status='pending',summary='Still queued at NCBI; open the result link later');return None


async def choose_structure(context):
    if context.get('structure_pdb'):
        pdb,seq,chain=structure_chain(context['structure_pdb'],context.get('structure_chain',''))
        origin = {'origin': 'experimental_structure' if context.get('pdb_id') else 'uploaded_structure',
                  'chain': chain, 'sequence_length': len(seq)}
        if context.get('pdb_id'):
            origin.update(pdb_id=context['pdb_id'], source_chain=context.get('source_chain', chain))
        return pdb, origin
    accession=context.get('accession','')
    if not accession:raise NotApplicable('Upload a PDB structure or choose a UniProt entry')
    # Fetch exactly mapped experimental chain, never a random chain of a complex.
    try:
        mapping=await get_json('https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/'+accession)
        for match in (mapping or {}).get(accession,[])[:2]:
            identity=match.get('pdb_id','');chain=match.get('chain_id','')
            if not re.fullmatch('[0-9A-Za-z]{4}',identity) or len(chain)!=1:continue
            raw=await request('GET','https://files.rcsb.org/download/'+identity.upper()+'.pdb',limit=2_000_000)
            if raw:
                pdb,seq,_=structure_chain(raw.decode(),chain)
                return pdb,{'origin':'experimental_structure','pdb_id':identity,'chain':chain,'coverage':match.get('coverage'),'resolution':match.get('resolution'),'sequence_length':len(seq),'matches_reference_sequence':seq==context.get('sequence')}
    except (ServiceError,ValueError,httpx.HTTPError):pass
    models=await get_json('https://alphafold.ebi.ac.uk/api/prediction/'+accession)
    model=next((m for m in (models or []) if m.get('uniprotAccession')==accession),None)
    if model and re.fullmatch(r'https://alphafold\.ebi\.ac\.uk/files/[A-Za-z0-9_.-]+\.pdb',model.get('pdbUrl','')):
        raw=await request('GET',model['pdbUrl'],limit=2_000_000)
        pdb,seq,chain=structure_chain(raw.decode(),model.get('chainId',''))
        return pdb,{'origin':'predicted_structure','model':model.get('modelEntityId'),'mean_plddt':model.get('globalMetricValue'),'chain':chain,'sequence_length':len(seq),'matches_reference_sequence':seq==context.get('sequence')}
    raise NotApplicable('No mapped experimental or predicted structure was found')


def foldseek_hits(value, include_alignments=False):
    rows=[]
    for result in value.get('results',[]):
        per_database=[]
        alignments=result.get('alignments',[])
        groups=alignments if alignments and isinstance(alignments[0],list) else [alignments]
        for group in groups:
            for hit in group:
                if not isinstance(hit,dict):continue
                if len(per_database)<20:
                    per_database.append({'database':result.get('db'),**{k:hit.get(k) for k in ('target','description','seqId','eval','score','qStartPos','qEndPos','dbStartPos','dbEndPos','qLen','dbLen','alnLength','prob','qtmscore','ttmscore','taxId','taxName', *(['qAln','dbAln'] if include_alignments else [])) if k in hit}})
        rows.extend(per_database)
    return rows


async def foldseek(context,row,emit):
    row.update(evidence_level='structure_similarity',mode='3diaa')
    if not row.get('job_id'):
        if row.get('submission_started'):raise ServiceError('Previous submission outcome unknown; no duplicate job submitted')
        pdb,origin=await choose_structure(context);row['query_structure']=origin
        data=await get_json('https://search.foldseek.com/api/databases')
        available={d['path']:d for d in (data or {}).get('databases',[]) if d.get('status')=='COMPLETE' and not any(d.get(k) for k in ('motif','rna'))}
        chosen=[k for k in ('pdb100','afdb-swissprot') if k in available]
        if not chosen:raise ServiceError('The selected Foldseek databases are not available')
        row['databases']=[{'id':k,'version':available[k].get('version')} for k in chosen]
        row['submission_started']=True;emit()
        raw=await request('POST','https://search.foldseek.com/api/ticket',data={'q':pdb,'database[]':chosen,'mode':'3diaa'},submit=True)
        ticket=json.loads(raw);identity=ticket.get('id','')
        if not re.fullmatch('[A-Za-z0-9_-]{8,100}',identity):raise ServiceError('Foldseek did not return a job ID')
        row.update(job_id=identity,status='waiting',url='https://search.foldseek.com/result/'+identity+'/0');emit()
        if context.get('submit_only'):
            row.update(status='pending',next_poll=time.time()+10,summary='Submitted to Foldseek; check using job_id');return None
    for _ in range(context.get('poll_attempts',36)):
        status=await get_json('https://search.foldseek.com/api/ticket/'+row['job_id'])
        if (status or {}).get('status')=='COMPLETE':
            result=await get_json('https://search.foldseek.com/api/result/'+row['job_id']+'/0',{} if context.get('include_alignments') else {'format':'brief'})
            if not isinstance(result,dict) or 'results' not in result:raise ServiceError('Foldseek returned an incomplete result')
            hits=foldseek_hits(result or {}, bool(context.get("include_alignments")))
            return {'hits':hits,'count':len(hits),'retention':'Up to 20 hits per database; see server link for all hits. seqId is percent identity.','mode':'3diaa','query_structure':row.get('query_structure'),
                    'databases':row.get('databases'),'interpretation':'Structural similarity is not evidence of binding, identical function or experimentally validated activity.'}
        if (status or {}).get('status') in {'ERROR','UNKNOWN','RATELIMIT'}:raise ServiceError('Foldseek job is unavailable')
        row.update(status='waiting',summary='Foldseek is searching public structure databases');emit()
        if context.get('poll_attempts')!=1:await asyncio.sleep(10)
    row.update(next_poll=time.time()+10,status='pending',summary='Still queued at Foldseek; open the result link later');return None


async def annotation(service,context,row):
    accession=context.get('accession','')
    if not ACCESSION.fullmatch(accession):raise NotApplicable('A matched UniProt accession is required')
    row['evidence_level']='database_annotation'
    if service=='kegg':
        from mcp_tools.tools.database.kegg.query import protein_annotations
        data = await protein_annotations(context)
        if data['genes']: row['url'] = data['genes'][0]['url']
        return data
    if service=='interpro':
        url='https://www.ebi.ac.uk/interpro/api/entry/interpro/protein/uniprot/'+accession+'/'
        obj=await get_json(url,{'page_size':25}) or {};rows=obj.get('results',[])[:25]
        row['url']='https://www.ebi.ac.uk/interpro/protein/UniProt/'+accession+'/'
        return {'count':len(rows),'total':obj.get('count',0),'entries':rows,'truncated':bool(obj.get('next'))}
    if service=='quickgo':
        obj=await get_json('https://www.ebi.ac.uk/QuickGO/services/annotation/search',{'geneProductId':'UniProtKB:'+accession,'limit':25}) or {}
        rows=[{k:v for k,v in a.items() if k in {'goId','goName','goEvidence','qualifier','evidenceCode','reference','taxonId','assignedBy','symbol','withFrom'}} for a in obj.get('results',[])[:25]]
        row['url']='https://www.ebi.ac.uk/QuickGO/annotations?geneProductId='+accession
        return {'count':len(rows),'total':obj.get('numberOfHits',0),'annotations':rows,'truncated':obj.get('numberOfHits',0)>len(rows)}
    if service=='reactome':
        rows=await get_json('https://reactome.org/ContentService/data/mapping/UniProt/'+accession+'/pathways') or []
        row['url']='https://reactome.org/ContentService/data/mapping/UniProt/'+accession+'/pathways'
        return {'count':min(len(rows),25),'total':len(rows),'pathways':[{k:v for k,v in x.items() if k in {'stId','displayName','speciesName','isInferred','isInDisease'}} for x in rows[:25]]}
    if service=='string':
        taxon=context.get('entry',{}).get('organism',{}).get('taxonId')
        if not taxon:raise NotApplicable('The selected protein needs an identified organism')
        params={'identifiers':accession,'species':int(taxon),'limit':10,'required_score':700,'caller_identity':'VenusFactory2'}
        rows=await get_json('https://string-db.org/api/json/interaction_partners',params) or []
        row.update(url='https://string-db.org/cgi/network?'+urlencode({'identifiers':accession,'species':taxon}),evidence_level='functional_association')
        return {'count':min(len(rows),10),'species':taxon,'partners':rows[:10],'interpretation':'STRING scores combine evidence channels and may represent functional association, not direct physical binding.'}
    if service=='alphafold':
        rows=await get_json('https://alphafold.ebi.ac.uk/api/prediction/'+accession) or []
        row.update(url='https://alphafold.ebi.ac.uk/entry/'+accession,evidence_level='model_prediction')
        keep={'modelEntityId','uniprotAccession','sequenceStart','sequenceEnd','latestVersion','globalMetricValue','fractionPlddtVeryLow','fractionPlddtLow','pdbUrl','cifUrl','paeDocUrl','toolUsed'}
        return {'count':min(len(rows),5),'models':[{k:v for k,v in x.items() if k in keep} for x in rows[:5]],'interpretation':'Predicted structures and pLDDT are not experimental structure or binding validation.'}
    if service=='pdbe':
        obj=await get_json('https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/'+accession) or {};rows=obj.get(accession,[])
        row.update(url='https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/'+accession,evidence_level='experimental_structure_metadata')
        return {'count':min(len(rows),20),'total':len(rows),'structures':rows[:20],'truncated':len(rows)>20}
    raise ServiceError('Unknown protein service')


async def run_service(service, context, update=lambda _:None, cache_dir=None):
    from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed
    assert_tool_call_allowed('query_public_protein_service', {'service':service})
    if service == 'blast': assert_tool_call_allowed('query_ncbi_blast')
    if service == 'foldseek_remote': assert_tool_call_allowed('query_foldseek_remote')
    if service == 'interproscan': assert_tool_call_allowed('query_interproscan')
    if service == 'kegg': assert_tool_call_allowed('query_kegg')
    if service == 's4pred': assert_tool_call_allowed('predict_s4pred')
    if service not in SERVICE_IDS:raise ServiceError('Unknown public protein service')
    spec=next(r for r in CATALOG if r[0]==service);key=digest_input(service,context)
    path=Path(cache_dir)/(service+'-'+key+'.json') if cache_dir else None
    row={'id':service,'label':spec[1],'label_zh':spec[2],'status':'running','url':spec[4],'started':time.time(),'input_sha256':key}
    if context.get('remote_job_id'):
        identity=context['remote_job_id']
        row.update(job_id=identity,url=('https://blast.ncbi.nlm.nih.gov/Blast.cgi?'+urlencode({'CMD':'Get','RID':identity})) if service=='blast' else ('https://search.foldseek.com/result/'+identity+'/0' if service=='foldseek_remote' else spec[4]))
    if path and path.exists():
        row=json.loads(path.read_text())
        if row.get('status') in {'completed','empty'}:
            row.pop('summary',None)
            update(row);return row
        row.update(status='running');row.pop('summary',None);row.pop('finished',None)
    def emit():
        save(path,row);update(dict(row))
    emit()
    try:
        if service=='blast':task=blast(context,row,emit)
        elif service=='foldseek_remote':task=foldseek(context,row,emit)
        elif service=='interproscan':
            from .interproscan import run as scan
            task=scan(context,row,emit)
        elif service=='s4pred':
            from mcp_tools.models.pretrained.s4pred import predict
            from backend.web_v2.job_queue import run_in_slot
            task=asyncio.to_thread(run_in_slot,'cpu',predict,context.get('sequence',''))
        else:task=annotation(service,context,row)
        data=await asyncio.wait_for(task,timeout=420 if service in {'blast','foldseek_remote','interproscan','s4pred'} else 80)
        if data is not None:
            row.update(data=data,count=data.get('count',0),status='completed' if data.get('count') else 'empty')
            row.pop('summary',None)
    except asyncio.CancelledError:
        row.update(status='cancelled',summary='Local monitoring stopped');emit();raise
    except NotApplicable as exc:row.update(status='not_applicable',summary=str(exc))
    except asyncio.TimeoutError:row.update(status='pending' if row.get('job_id') else 'unavailable',summary='Service wait limit reached')
    except Exception as exc:row.update(status='unavailable',summary=str(exc) if isinstance(exc,ServiceError) else 'Public service temporarily unavailable')
    row['finished']=time.time();emit();return row


def source(row):
    if not row.get('data'):return None
    return {'title':row['label']+' · public analysis','url':row['url'],'kind':'public_service','service':row['id'],
            'evidence_level':row.get('evidence_level','database_annotation'),
             'text':json.dumps({'service':row['label'],'status':row['status'],'job_id':row.get('job_id'),'data':row['data']},ensure_ascii=False)}


def report_appendix(rows, lang='en'):
    """Keep actual server results in downloads even when a writer omits them."""
    if not rows:return ''
    zh=lang=='zh'
    def cell(value):
        return str(value if value is not None else '—').replace('|','/').replace('\n',' ').replace('<','&lt;').replace('>','&gt;')
    parts=['## '+('公共服务分析结果' if zh else 'Public service results')]
    for row in rows:
        parts.append('### '+row['label'])
        parts.append(' · '.join([cell(row['status']),str(row.get('count',0))+' results', '[Open result]('+row['url']+')']))
        if row.get('job_id'):parts.append('Job: `'+cell(row['job_id'])+'`')
        if row.get('summary'):parts.append(cell(row['summary']))
        data=row.get('data',{})
        if data.get('interpretation'):parts.append(data['interpretation'])
        if data.get('query_structure'):parts.append('Query structure: '+cell(json.dumps(data['query_structure'],ensure_ascii=False)))
        hits=data.get('hits',[])
        if hits:
            keys=['accession','title','evalue','identity_pct','query_coverage_pct'] if row['id']=='blast' else ['database','target','description','eval','seqId','score']
            parts.append('| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(cell(h.get(k)) for k in keys)+' |' for h in hits))
        elif data:
            # Full bounded structured results are also available in Evidence JSON.
            parts.append('```json\n'+json.dumps(data,ensure_ascii=False,indent=2).replace('```','` ` `')+'\n```')
    return '\n\n'.join(parts)
