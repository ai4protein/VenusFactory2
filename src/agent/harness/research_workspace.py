"""Goal-scoped files, isolated Python execution and reusable program receipts."""
from __future__ import annotations

from mcp_tools.tools.capability_policy import guard_operation
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import re
import uuid

from .contracts import digest, file_hash
from .host import write_json
from utils.scoped_files import open_file, read_bytes, write_bytes, copy_file, parts

LIBRARY = Path(__file__).resolve().parents[3] / 'research_runs/rsi/research_capabilities'

def safe_file(root, name, *, exists=True):
    parts(name)
    if not isinstance(name, str) or not name or Path(name).is_absolute() or any(p in {'..', ''} or p.startswith('.') for p in Path(name).parts):
        raise ValueError('Use a relative workspace filename')
    root = Path(root).resolve()
    path = root / name
    path.resolve().relative_to(root)
    if any(p.is_symlink() for p in [path, *path.parents] if p != root and root in p.parents):
        raise ValueError('Symlinks are unavailable')
    if exists and (not path.is_file() or path.stat().st_size > 100 * 1024**2):
        raise ValueError('Workspace file missing or too large')
    if exists:
        with open_file(root, name, max_bytes=100*1024**2): pass
    return path

def workspace_hash(root, name):
    return hashlib.sha256(read_bytes(root,name)).hexdigest()

async def _run_python(workspace, control, script, timeout=120, *, on_progress=None):
    """Use the same isolated execution service as Agent session coding."""
    from .session_sandbox import execute
    safe_file(workspace, script)
    result = await execute(workspace, control, 'python', script, timeout, on_progress=on_progress)
    result['script_sha256'] = workspace_hash(workspace, script)
    return result

async def run_python(workspace, control, script, timeout=120, *, on_progress=None):
    from backend.web_v2.job_queue import resource_slot
    def waiting(job):
        if on_progress:on_progress({'status':'queued','tool':'run_research_python','script':script,'elapsed_seconds':0,'stdout_tail':'','stderr_tail':'','queue':job})
    async with resource_slot('cpu',kind='research:python',on_wait=waiting):
        return await _run_python(workspace,control,script,timeout,on_progress=on_progress)

class ResearchWorkspace:
    def __init__(self, workspace, control, plan, *, runner=run_python, development_root=None):
        self.workspace=Path(workspace);self.control=Path(control);self.plan=plan;self.runner=runner
        self.workspace.mkdir(parents=True,exist_ok=True);self.control.mkdir(parents=True,exist_ok=True)
        self.receipts=[];self.checkpoint=None;self.proposals=[]
        from .research_development import Development
        self.development=Development(development_root or self.control/'development')

    def record(self, tool, args, result):
        row={'tool':tool,'arguments_sha256':digest(args),'result':result}
        self.receipts.append(row)
        write_json(self.control/'receipts.json',self.receipts)
        return result

    async def list_research_files(self) -> dict:
        """List files in the current research workspace."""
        from backend.web_v2.research_files import list_files
        return {'files':[{'path':row['rel'],'bytes':row['size']} for row in list_files({'workspace':self.workspace},limit=500)['files']]}

    async def read_research_file(self, path: str) -> dict:
        """Read text, CSV, molecular structure text or up to 30 pages from a PDF."""
        p=Path(path)
        if p.suffix.lower()=='.pdf':
            import io
            data=read_bytes(self.workspace,path,max_bytes=20*1024**2)
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader
            reader=PdfReader(io.BytesIO(data))
            return {'path':path,'pages':len(reader.pages),'text':'\n'.join(page.extract_text() or '' for page in reader.pages[:30])[:200000],'sha256':hashlib.sha256(data).hexdigest()}
        data=read_bytes(self.workspace,path,max_bytes=200000)
        return {'path':path,'text':data.decode(),'sha256':hashlib.sha256(data).hexdigest()}

    async def write_research_file(self, path: str, content: str) -> dict:
        """Create or revise Python code, a report, or a small data file in this workspace."""
        if len(content.encode())>200000:raise ValueError('File is too large')
        data=content.encode();write_bytes(self.workspace,path,data)
        return self.record('write_research_file',{'path':path,'content':content},{'path':path,'sha256':hashlib.sha256(data).hexdigest()})

    @guard_operation('run_research_python')
    async def run_research_python(self, path: str, timeout_seconds: int=120) -> dict:
        """Execute saved Python in an isolated, memory-bounded single process without networking or subprocesses. Scientific Python packages are available. Use n_jobs=1; model training needing subprocesses uses queued MCP tools. Save relative outputs; inspect exit_code."""
        source_hash=workspace_hash(self.workspace,path)
        kwargs={'on_progress':lambda value:write_json(self.control/'execution.json',progress_view(value))} if self.runner is run_python else {}
        result=await self.runner(self.workspace,self.control/'python',path,timeout_seconds,**kwargs)
        if result['script_sha256'] != source_hash:raise ValueError('Script changed during execution')
        result['submitted_source_sha256']=source_hash
        return self.record('run_research_python',{'path':path,'timeout':timeout_seconds},result)

    @guard_operation('propose_research_capability')
    async def propose_research_capability(self, name: str, description: str, source: str, files: list[str], test_inputs: list[dict], review_spec: dict | None = None) -> dict:
        """Propose reusable code after implementing run(inputs: dict) -> JSON in the source module. Include any weights in files. The host replays test_inputs in fresh isolated processes; scientific accuracy must be evaluated separately. Research Critic and separate public admission Review are required. Call get_research_review_requirements first. Supply review_spec and private evidence files; only runtime files become public."""
        if not name.strip() or not description.strip() or not 1<=len(test_inputs)<=8 or len(files)>20:
            raise ValueError('Provide a name, scope, files and 1..8 test inputs')
        if len(self.proposals)>=2: raise ValueError('Submit at most two candidates per iteration')
        if self.plan.get('deliverable')=='model' and (review_spec or {}).get('kind')!='model':
            raise ValueError('This objective requires a fitted model, evaluation evidence and a model delivery')
        paths=sorted(set([source,*files]))
        if any(Path(p).name in {'manifest.json','admission.json','origin.json','capability_input.json','capability_result.json','capability_runner.py'} for p in paths): raise ValueError('Reserved runtime filename')
        bundle=self.control/'proposals'/uuid.uuid4().hex;bundle.mkdir(parents=True)
        for path in paths:
            copy_file(self.workspace,path,bundle,path)
        manifest={'name':name[:120],'description':description[:2000],'source':source,'files':{p:file_hash(bundle/p) for p in paths},'tests':test_inputs}
        if review_spec is not None:
            from .capability_review import validate_spec
            validate_spec(review_spec)
            evidence=review_spec['evidence_files']
            if set(evidence)&set(paths): raise ValueError('Keep private evidence separate from runtime files')
            for name in evidence:
                if Path(name).name in {'manifest.json','admission.json','origin.json'}: raise ValueError('Reserved evidence filename')
                copy_file(self.workspace,name,bundle,name)
            manifest.update(review_spec=review_spec,evidence_files={p:file_hash(bundle/p) for p in evidence})
            experiment_id=review_spec.get('development_experiment_id')
            if self.plan.get('deliverable')=='model' and not experiment_id:
                raise ValueError('Evaluate a baseline and revised generation, then set development_experiment_id')
            if experiment_id:
                if review_spec['kind']!='model':raise ValueError('Development experiments describe models')
                data=json.loads(read_bytes(bundle,review_spec['evaluation_file'],max_bytes=20*1024**2))
                manifest['development']=self.development.selected(experiment_id,manifest,data)
        results=[]
        for inputs in test_inputs:
            one=await invoke_bundle(bundle,manifest,inputs,self.control/'verification',runner=self.runner)
            two=await invoke_bundle(bundle,manifest,inputs,self.control/'verification',runner=self.runner)
            if digest(one)!=digest(two):raise ValueError('Fresh-process replay differed; set deterministic seeds')
            results.append({'input_sha256':digest(inputs),'output_sha256':digest(one)})
        manifest['validation']={'kind':'execution_replay','results':results}
        manifest['id']=digest(manifest)
        write_json(bundle/'manifest.json',manifest)
        self.proposals.append(bundle)
        return self.record('propose_research_capability',{'source':source,'tests':test_inputs},{'id':manifest['id'],'name':name,'validation':manifest['validation']})

    async def submit_research_checkpoint(self, summary: str, completed_steps: list[str], artifacts: list[str], needs_input: list[str]) -> dict:
        """Checkpoint actual work for the Scientific Critic. Cite existing result files, and only step IDs from the current plan. Use needs_input for specific missing user data. This does not itself complete the goal."""
        ids={s['id'] for s in self.plan['steps']}
        if not summary.strip() or not set(completed_steps)<=ids or (completed_steps and not artifacts):raise ValueError('Cite current plan steps and actual artifacts')
        saved=[]
        for name in artifacts:
            data=read_bytes(self.workspace,name);saved.append({'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)})
        self.checkpoint={'summary':summary[:6000],'completed_steps':completed_steps,'artifacts':saved,'needs_input':needs_input}
        write_json(self.control/'checkpoint.json',self.checkpoint)
        return {'checkpointed':True,'artifacts':saved}

    async def get_research_review_requirements(self) -> dict:
        """Get the interface, evaluation and independent Review requirements for reusable tools/models."""
        from .capability_review import requirements
        return requirements()

    async def get_research_strategy_guide(self) -> dict:
        """Get model-development strategies, evidence requirements and measured prior generations. Supports custom algorithms, few-shot, semi-supervised and meta-learning; PLM training is optional."""
        from .research_development import guide
        return {**guide(),'development':self.development.summary()}

    @guard_operation('evaluate_research_model')
    async def evaluate_research_model(self, name: str, source: str, files: list[str], evaluation_file: str, prediction_key: str, hypothesis: str, strategy: str='custom', parent_experiment_id: str='', variant: str='candidate') -> dict:
        """Snapshot a fitted run(inputs) model, replay validation predictions in isolated processes and score on the host. First use variant=baseline; revised candidates cite parent_experiment_id and identical data/splits. Never use final test scores to choose a generation. Call get_research_strategy_guide for the protocol."""
        args=dict(name=name,source=source,files=files,evaluation_file=evaluation_file,prediction_key=prediction_key,hypothesis=hypothesis,strategy=strategy,parent_experiment_id=parent_experiment_id,variant=variant)
        result=await self.development.evaluate(self.workspace,self.runner,**args)
        return self.record('evaluate_research_model',args,result)

    def workspace_tools(self):
        from langchain_core.tools import StructuredTool
        from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
        from mcp_tools.tools.capability_policy import guard_tool
        from mcp_tools.tools.tool_surface import WORKSPACE_ALIASES
        return [guard_tool(StructuredTool.from_function(coroutine=getattr(self, method), name=name,
                    metadata={'workspace_relative_paths': True}))
                for method, name in WORKSPACE_ALIASES.items() if is_tool_allowed(name)]

    def tools(self):
        from langchain_core.tools import StructuredTool
        from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
        from mcp_tools.tools.capability_policy import guard_tool
        return self.workspace_tools() + [guard_tool(StructuredTool.from_function(coroutine=getattr(self,name),name=name,metadata={"workspace_relative_paths": True})) for name in ['get_research_strategy_guide','evaluate_research_model','get_research_review_requirements','propose_research_capability','submit_research_checkpoint'] if is_tool_allowed(name)]

def log_tail(path, limit=10000):
    with Path(path).open('rb') as f:
        f.seek(max(0,Path(path).stat().st_size-limit));return f.read(limit).decode(errors='replace')

def progress_view(value):
    """Progress is reported only when the program supplies measured counters."""
    for line in (value.get('stdout_tail','')+'\n'+value.get('stderr_tail','')).splitlines()[-80:]:
        try:
            row=json.loads(line)
            if isinstance(row,dict) and row.get('event')=='progress' and type(row.get('step')) in (int,float) and type(row.get('total')) in (int,float) and 0<=row['step']<=row['total'] and row['total']>0:
                value['progress']={'step':row['step'],'total':row['total'],'metrics':row.get('metrics',{})}
        except ValueError:
            match=re.search(r'\bepoch\s+(\d+)\s*/\s*(\d+)',line,re.I)
            if match and 0<int(match[2]) and int(match[1])<=int(match[2]):value['progress']={'step':int(match[1]),'total':int(match[2]),'metrics':{}}
    return value

async def invoke_bundle(bundle, manifest, inputs, control, *, runner=run_python, batch=False):
    bundle,control=Path(bundle),Path(control);work=control/uuid.uuid4().hex;work.mkdir(parents=True)
    for name,sha in manifest['files'].items():
        p=safe_file(bundle,name)
        if file_hash(p)!=sha:raise ValueError('Registered program changed')
        target=work/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    write_json(work/'capability_input.json',inputs)
    (work/'capability_runner.py').write_text('import importlib.util,json\nspec=importlib.util.spec_from_file_location("candidate",'+repr(manifest['source'])+')\nm=importlib.util.module_from_spec(spec)\nspec.loader.exec_module(m)\ninputs=json.load(open("capability_input.json"))\nresult='+('[m.run(row) for row in inputs]' if batch else 'm.run(inputs)')+'\nwith open("capability_result.json","w") as f: json.dump(result,f,allow_nan=False)\n')
    result=await runner(work,control/'runs','capability_runner.py',120)
    if result['exit_code']!=0:raise ValueError('Capability execution failed: '+result['stderr'][-2000:])
    value=json.loads(read_bytes(work,'capability_result.json',max_bytes=2000000));json.dumps(value,allow_nan=False);return value

def publish_proposals(proposals, goal_id, library=LIBRARY):
    from .capability_review import admitted
    from .capability_catalog import register
    library=Path(library);library.mkdir(parents=True,exist_ok=True);published=[]
    for bundle in proposals:
        bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text());ident=manifest['id']
        if digest({k:v for k,v in manifest.items() if k!='id'})!=ident:raise ValueError('Capability manifest changed')
        receipt=admitted(bundle,manifest)
        dest=library/ident
        if not dest.exists():
            stage=library/('staging_'+uuid.uuid4().hex);stage.mkdir()
            for name,sha in manifest['files'].items():
                p=safe_file(bundle,name)
                if file_hash(p)!=sha:raise ValueError('Candidate runtime changed')
                target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
            write_json(stage/'manifest.json',manifest);write_json(stage/'admission.json',receipt)
            write_json(stage/'origin.json',{'goal_id':goal_id,'kind':'autonomous_research'})
            stage.rename(dest)
        published.append(register(library,manifest,receipt,goal_id,visibility='private' if library.name=='private_capabilities' else 'public'))
    return published

def capability_cards(library=LIBRARY):
    from .capability_catalog import cards
    return cards(library)

def search_capabilities(query='', limit=10, library=LIBRARY, offset=0):
    from .capability_catalog import search
    from mcp_tools.tools.capability_policy import CapabilityStore
    from .capability_catalog import cards
    settings=CapabilityStore()
    values=[c for c in cards(library) if settings.enabled('model:capability:'+c['id']) and all(w in json.dumps(c,ensure_ascii=False).casefold() for w in query.casefold().split())]
    return {'total':len(values),'items':values[max(0,int(offset)):max(0,int(offset))+max(1,min(int(limit),50))]}

def admitted_manifest(capability_id, library):
    from .capability_review import admitted
    from .capability_catalog import require_registered
    if not re.fullmatch('[a-f0-9]{64}',capability_id):raise ValueError('Invalid capability ID')
    bundle=Path(library)/capability_id;manifest=json.loads(safe_file(bundle,'manifest.json').read_text())
    if digest({k:v for k,v in manifest.items() if k!='id'})!=capability_id:raise ValueError('Capability manifest changed')
    receipt=admitted(bundle,manifest);require_registered(library,capability_id,receipt['receipt_sha256'])
    for name,sha in manifest['files'].items():
        if file_hash(safe_file(bundle,name))!=sha:raise ValueError('Capability source changed')
    return bundle,manifest,receipt

def describe_capability(capability_id, library=LIBRARY):
    _,manifest,receipt=admitted_manifest(capability_id,library)
    spec=manifest['review_spec']
    return {**{k:manifest[k] for k in ['id','name','description','tests','validation']},
            **{k:spec[k] for k in ['kind','author','version','scope','input_schema','output_schema','dependencies']},
            'review_status':'approved','review':receipt}

@guard_operation('run_generated_capability')
async def run_capability(capability_id, inputs, *, library=LIBRARY):
    from .capability_review import schema_validator
    bundle,manifest,_=admitted_manifest(capability_id,library)
    schema_validator(manifest['review_spec']['input_schema']).validate(inputs)
    result=await invoke_bundle(bundle,manifest,inputs,Path(library)/'invocations'/capability_id)
    schema_validator(manifest['review_spec']['output_schema']).validate(result)
    return result
