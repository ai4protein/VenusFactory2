"""Owner-scoped report materials, bounded extraction and immutable job snapshots."""
from pathlib import Path
import asyncio
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import uuid

EXTENSIONS={'.pdf','.csv','.tsv','.xlsx','.txt','.md','.json'}
MAX_TEXT=120000

def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()

def extract(path):
    """Run in a bounded child, not in an API worker. Never execute spreadsheet formulas."""
    path=Path(path);ext=path.suffix.lower();chunks=[];size=0;truncated=False;units=0
    def add(text):
        nonlocal size,truncated
        text=str(text)
        if size+len(text)>MAX_TEXT:truncated=True
        chunks.append(text[:max(0,MAX_TEXT-size)]);size+=len(chunks[-1])
        return size<MAX_TEXT
    if ext=='.pdf':
        from pypdf import PdfReader
        reader=PdfReader(path);units=len(reader.pages);truncated=units>30
        for i,page in enumerate(reader.pages[:30]):
            text=page.extract_text() or ''
            if not add(f'\n[Page {i+1}]\n'+text):break
        if not any(re.sub(r'\[Page \d+\]','',s).strip() for s in chunks):raise ValueError('This PDF has no extractable text. Upload a text/OCR version.')
    elif ext=='.xlsx':
        from openpyxl import load_workbook
        book=load_workbook(path,read_only=True,data_only=True)
        try:
            units=len(book.sheetnames);truncated=units>12
            for sheet in book.worksheets[:12]:
                add('\n[Sheet '+sheet.title+']\n')
                for i,row in enumerate(sheet.iter_rows(values_only=True)):
                    if i>=300:truncated=True;break
                    if not add(str(i+1)+'\t'+'\t'.join('' if x is None else str(x) for x in row[:60])+'\n'):break
                    if len(row)>60:truncated=True
                if size>=MAX_TEXT:break
        finally:book.close()
    elif ext in {'.csv','.tsv'}:
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for i,row in enumerate(csv.reader(stream,delimiter='\t' if ext=='.tsv' else ',')):
                if i>=500:truncated=True;break
                units+=1
                if not add(f'[Row {i+1}] '+json.dumps(row[:60],ensure_ascii=False)+'\n'):break
                if len(row)>60:truncated=True
    elif ext in EXTENSIONS:
        with path.open(encoding='utf-8-sig') as stream:
            text=stream.read(MAX_TEXT+1);add(text)
        units=len(''.join(chunks).splitlines())
    else:raise ValueError('Upload PDF, CSV, TSV, XLSX, TXT, MD or JSON')
    text=''.join(chunks)
    if not text.strip():raise ValueError('The material contains no readable text')
    return {'text':text,'extraction':{'format':ext[1:],'units':units,'truncated':truncated or size>=MAX_TEXT,'characters':len(text)}}

async def extract_isolated(path):
    proc=await asyncio.create_subprocess_exec(sys.executable,'-I',__file__,'extract',str(path),stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
    try:
        out,_=await asyncio.wait_for(proc.communicate(),45)
        data=json.loads(out)
        if proc.returncode or 'error' in data:raise ValueError(data.get('error','Could not read this material'))
        return data
    except (asyncio.TimeoutError,json.JSONDecodeError):raise ValueError('Material extraction exceeded its processing limit; supply a smaller text extract') from None
    finally:
        if proc.returncode is None:proc.kill();await proc.wait()

def owner_folder(root,owner): return Path(root)/'.materials'/hashlib.sha256(owner.encode()).hexdigest()

def resolve(root,owner,identity):
    if not isinstance(identity,str) or not re.fullmatch('[a-f0-9]{32}',identity):raise FileNotFoundError('Material not found')
    folder=owner_folder(root,owner)/identity
    row=json.loads((folder/'metadata.json').read_text())
    path=folder/('source.'+row['extraction']['format'])
    if path.is_symlink() or sha(path)!=row['sha256']:raise ValueError('Material changed; upload it again')
    return row,path

def card(row):return {k:row[k] for k in ['id','name','bytes','sha256','extraction']}

async def upload(root,owner,file):
    from backend.web_v2.upload_limits import save_upload
    name=Path((file.filename or 'material').replace('\\','/')).name[:160]
    ext=Path(name).suffix.lower()
    if ext not in EXTENSIONS:raise ValueError('Upload PDF, CSV, TSV, XLSX, TXT, MD or JSON')
    identity=uuid.uuid4().hex;folder=owner_folder(root,owner)/identity
    folder.mkdir(parents=True,mode=0o700);path=folder/('source'+ext)
    try:
        size=await save_upload(file,path);path.chmod(0o600)
        data=await extract_isolated(path)
        row={'id':identity,'name':name,'bytes':size,'sha256':await asyncio.to_thread(sha,path),**data}
        from backend.web_v2.research_jobs import write
        write(folder/'metadata.json',row)
        return card(row)
    except BaseException:
        # Only this request's newly created folder can be removed.
        shutil.rmtree(folder);raise

def sources(root,owner,identities,job_id):
    if not isinstance(identities,list) or len(identities)>12 or len(set(identities))!=len(identities):raise ValueError('Attach at most 12 distinct materials')
    rows=[]
    for identity in identities:
        row,_=resolve(root,owner,identity)
        rows.append({**card(row),'title':row['name'],'text':row['text'],'kind':'user_material','evidence_level':'user_supplied_material',
                     'url':f'/api/report/jobs/{job_id}/materials/{identity}'})
    return rows

def snapshot(root,owner,job_id,rows):
    folder=Path(root)/job_id/'materials';folder.mkdir(parents=True,exist_ok=True,mode=0o700)
    for row in rows:
        _,path=resolve(root,owner,row['id']);dest=folder/(row['id']+'.'+row['extraction']['format'])
        shutil.copyfile(path,dest);dest.chmod(0o600)
        if sha(dest)!=row['sha256']:raise ValueError('Material changed while preparing research')

def download(service,owner,job_id,identity):
    job_id,row=service.owned(owner,job_id)
    material=next((r for r in row['payload'].get('material_sources',[]) if r['id']==identity),None)
    if not material:raise FileNotFoundError('Material not found')
    path=service.folder(job_id)/'materials'/(material['id']+'.'+material['extraction']['format'])
    if sha(path)!=material['sha256']:raise ValueError('Material snapshot changed')
    return path,material['name']

def chapter_materials(sources,index,terms):
    rows=[s for s in sources if s.get('kind')=='user_material']
    if not rows:return []
    # Rotate two materials across chapters so every attachment is read; add the
    # most relevant remaining material and prioritize matching paragraphs.
    keywords=set(terms.lower().split())
    score=lambda s:sum(s.get('text','').lower().count(w) for w in keywords)
    selected=[rows[(2*index)%len(rows)],rows[(2*index+1)%len(rows)],max(rows,key=score)]
    result=[]
    for row in {r['id']:r for r in selected}.values():
        text=row['text'];lines=text.splitlines();matches=[i for i,line in enumerate(lines) if any(w in line.lower() for w in keywords)]
        excerpt=text[:4000]
        for i in matches[:20]:excerpt+='\n'+ '\n'.join(lines[max(0,i-1):i+3])
        result.append({**row,'text':excerpt[:12000], 'excerpted':True})
    return result

if __name__=='__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
    resource.setrlimit(resource.RLIMIT_CPU,(35,35))
    try:print(json.dumps(extract(sys.argv[2]),ensure_ascii=False))
    except Exception as exc:print(json.dumps({'error':str(exc)[:200]}));sys.exit(1)
