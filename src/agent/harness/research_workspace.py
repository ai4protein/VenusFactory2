"""Goal-scoped files, isolated Python execution and reusable program receipts."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from mcp_tools.tools.capability_policy import guard_operation
from utils.scoped_files import copy_file, open_file, parts, read_bytes, replace_text, write_bytes

from .contracts import digest, file_hash
from .host import write_json
from .research_questions import validate_input_questions

ExperimentVariant = Annotated[
    Literal["auto", "baseline", "candidate", "ablation"],
    Field(description=(
        "auto starts a baseline without a parent; with parent_experiment_id, "
        "it evaluates a candidate revision. Use ablation for a controlled removal."
    )),
]

LIBRARY = Path(__file__).resolve().parents[3] / 'research_runs/rsi/research_capabilities'
RESEARCH_TOOLS = ('get_research_strategy_guide', 'get_research_baseline', 'prepare_research_dataset', 'validate_research_evaluation', 'train_research_model', 'evaluate_research_model', 'restore_research_evaluation', 'get_research_review_requirements', 'propose_research_capability', 'submit_research_checkpoint')

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
    def __init__(self, workspace, control, plan, *, runner=run_python, development_root=None, data_store=None):
        self.workspace=Path(workspace);self.control=Path(control);self.plan=plan;self.runner=runner
        self.workspace.mkdir(parents=True,exist_ok=True);self.control.mkdir(parents=True,exist_ok=True)
        self.receipts=[];self.checkpoint=None;self.proposals=[];self.latest_submission=None
        from .research_development import Development
        self.data_store=data_store
        self.development=Development(development_root or self.control/'development',data_store=data_store)

    def record(self, tool, args, result):
        row={'tool':tool,'arguments_sha256':digest(args),'result':result}
        self.receipts.append(row)
        write_json(self.control/'receipts.json',self.receipts)
        return result

    async def list_research_files(self) -> dict:
        """List files in the current research workspace."""
        from backend.web_v2.research_files import list_files
        return {'files':[{'path':row['rel'],'bytes':row['size']} for row in list_files({'workspace':self.workspace},limit=500)['files']]}

    async def read_research_file(self, path: str, offset_bytes: int=0, max_bytes: int=20000, view: Literal["auto", "text"]="auto") -> dict:
        """Preview a file. CSV/TSV defaults to columns and three sample rows; analyze full tables with isolated Python. Use view='text' for a targeted raw window. Text windows return next_offset_bytes; PDFs return up to 30 pages."""
        if view not in ("auto", "text"): raise ValueError("view must be auto or text")
        p=Path(path)
        if p.suffix.lower()=='.pdf':
            if offset_bytes: raise ValueError('Byte offsets apply to text files only')
            import io
            data=read_bytes(self.workspace,path,max_bytes=20*1024**2)
            try:
                from pypdf import PdfReader
            except ImportError:
                from PyPDF2 import PdfReader
            reader=PdfReader(io.BytesIO(data))
            return {'path':path,'pages':len(reader.pages),'text':'\n'.join(page.extract_text() or '' for page in reader.pages[:30])[:200000],'sha256':hashlib.sha256(data).hexdigest()}
        from .workspace_preview import file_preview
        return file_preview(self.workspace,path,offset_bytes=offset_bytes,max_bytes=max_bytes,view=view)

    async def write_research_file(self, path: str, content: str) -> dict:
        """Create or revise Python code, a report, or a small data file in this workspace."""
        if len(content.encode())>200000:raise ValueError('File is too large')
        data=content.encode();write_bytes(self.workspace,path,data)
        return self.record('write_research_file',{'path':path,'content':content},{'path':path,'sha256':hashlib.sha256(data).hexdigest()})

    async def edit_research_file(self, path: str, old_text: str, new_text: str, expected_sha256: str) -> dict:
        """Replace one exact occurrence in a saved UTF-8 file. Read it first; pass full-file sha256 or file_sha256, never slice_sha256. Use surrounding text to make the match unique. Changed files, missing or ambiguous matches are rejected; no code executes."""
        result = replace_text(self.workspace, path, old_text, new_text, expected_sha256)
        return self.record('edit_research_file', {'path':path, 'old_text':old_text,
            'new_text':new_text, 'expected_sha256':expected_sha256}, result)

    @guard_operation('run_research_python')
    async def run_research_python(self, path: str, timeout_seconds: int=120) -> dict:
        """Execute a saved .py script, not a data file, in an isolated, memory-bounded single process without networking or subprocesses. Scientific Python packages are available. Use n_jobs=1; model training needing subprocesses uses queued MCP tools. Save relative outputs; inspect exit_code."""
        from .session_sandbox import validate_execution
        validate_execution('python', path, timeout_seconds)
        source_hash=workspace_hash(self.workspace,path)
        kwargs={'on_progress':lambda value:write_json(self.control/'execution.json',progress_view(value))} if self.runner is run_python else {}
        result=await self.runner(self.workspace,self.control/'python',path,timeout_seconds,**kwargs)
        if result['script_sha256'] != source_hash:raise ValueError('Script changed during execution')
        result['submitted_source_sha256']=source_hash
        self.record('run_research_python',{'path':path,'timeout':timeout_seconds},result)
        from .execution_outcome import execution_error
        from .tool_diagnostics import bounded_message
        error=execution_error(result)
        if error:raise ValueError(bounded_message(error,self.workspace,1200))
        return result

    @guard_operation('propose_research_capability')
    async def propose_research_capability(self, name: str='', description: str='', source: str='', files: list[str] | None=None, test_inputs: list[dict] | None=None, review_spec: dict | None=None, proposal_file: str='') -> dict:
        """Submit a private candidate for SC scientific acceptance and independent technical Review. Prefer proposal_file: a workspace JSON with name, description, source, files, test_inputs and review_spec. Edit and resubmit that file after revision; unchanged metadata are preserved. Inline fields remain supported and save review/proposal.json. Read get_research_review_requirements first. Submission does not grant public release."""
        from .review_submission import ReviewSubmission
        submission=ReviewSubmission(self.workspace,self.control,
            dict(name=name,description=description,source=source,files=files,test_inputs=test_inputs,review_spec=review_spec),proposal_file)
        self.latest_submission=submission.record('preparing')
        try:
            result=await self._propose_research_capability(**submission.payload)
        except (ValueError,TypeError,OSError) as exc:
            self.latest_submission=submission.record('invalid',error=exc)
            raise ValueError(str(exc)+'; revise '+submission.path+' and resubmit using proposal_file') from exc
        self.latest_submission=submission.record('submitted',candidate_id=result['id'])
        write_json(self.proposals[-1]/'submission.json',self.latest_submission)
        return {**result,'proposal_file':submission.path,'submission_status':'awaiting_review'}

    async def _propose_research_capability(self, name, description, source, files, test_inputs, review_spec):
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
            for evidence_name in evidence:
                if Path(evidence_name).name in {'manifest.json','admission.json','origin.json'}: raise ValueError('Reserved evidence filename')
                if self.data_store and evidence_name == review_spec.get('evaluation_file'):
                    write_bytes(bundle,evidence_name,self.data_store.load_evaluation(evidence_name))
                else:
                    copy_file(self.workspace,evidence_name,bundle,evidence_name)
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
        """Checkpoint actual work for the Scientific Critic. Cite existing result files, and only step IDs from the current plan. Use needs_input only for specific missing user data. Use an empty list for unfinished work or an iteration budget limit; the goal loop continues automatically. This does not itself complete the goal."""
        validate_input_questions(needs_input)
        ids={s['id'] for s in self.plan['steps']}
        if not summary.strip():
            raise ValueError('Provide a brief summary of actual work before checkpointing')
        if not set(completed_steps)<=ids:
            raise ValueError('completed_steps accepts exact step IDs only, without titles or descriptions. '
                             'Current step IDs: '+json.dumps(sorted(ids))+
                             '. Put explanations in summary; do not mark unfinished steps complete.')
        if completed_steps and not artifacts:
            raise ValueError('Completed steps require existing artifact filenames. '
                             'Use completed_steps=[] for a partial checkpoint without completed work.')
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
        return {**guide(),'development':self.development.summary(),
                'reference_baselines':{'tool':'get_research_baseline','tasks':['regression','multilabel'],
                    'scope':'Fixed sequence controls for the declared task; inspect scope before choosing.'}}

    async def get_research_baseline(self, reference: str='', sequence_column: str='sequence', output_field: str='value', output_file: str='', task: str='regression') -> dict:
        """List classic sequence controls for task=regression or task=multilabel. Select a reference ID and output_file to save runnable fit/predict code with verified sequence encoding; omit output_file to read the source. Existing different files are never overwritten. Fit it with train_research_model, then develop a measured revision. Preserve its platform-reference provenance."""
        from .research_development import immutable_copy
        from .research_reference_baselines import reference_baseline
        result=reference_baseline(reference,sequence_column,output_field,task=task)
        if output_file:
            if not reference or not isinstance(output_file,str) or not output_file.endswith('.py'):
                raise ValueError('Select a reference ID and a relative .py output_file')
            safe_file(self.workspace,output_file,exists=False)
            immutable_copy(self.workspace,output_file,result['source'].encode())
            result={key:value for key,value in result.items() if key!='source'}
            result.update(source_file=output_file,
                next_step='Call train_research_model with this source_file, the prepared evaluation_file and variant=baseline. '
                'Then develop and measure a candidate using its parent_experiment_id; preserve the reference provenance.')
        return self.record('get_research_baseline',{'reference':reference,'sequence_column':sequence_column,
            'output_field':output_field,'output_file':output_file,'task':task},result)

    async def restore_research_evaluation(self, experiment_id: str, output_file: str='') -> dict:
        """Export the measured generation's frozen evaluation to a new session file for final Review. Never overwrite different existing data. Use the returned path in review_spec.evaluation_file and evidence_files; keep targets and splits unchanged."""
        result=self.development.export_evaluation(self.workspace,experiment_id,output_file)
        return self.record('restore_research_evaluation',{'experiment_id':experiment_id,'output_file':output_file},result)

    @guard_operation('validate_research_evaluation')
    async def validate_research_evaluation(self, evaluation_file: str, strategy: str='supervised') -> dict:
        """Check a dataset evaluation JSON before training: split sizes, finite targets and disjoint IDs, groups and exact inputs. Returns counts and file hashes, never labels or sequences. Use get_research_review_requirements for the format."""
        from .research_development import inspect_evaluation
        result=inspect_evaluation(self.workspace,evaluation_file,strategy,data_store=self.data_store)
        return self.record('validate_research_evaluation',
                           {'evaluation_file':evaluation_file,'strategy':strategy},result)

    @guard_operation('prepare_research_dataset')
    async def prepare_research_dataset(self, data_file: str, target_column: str, input_columns: list[str],
            group_column: str, provenance: str, grouping: str, id_column: str='', task: str='regression',
            seed: int=0, validation_fraction: float=0.15, test_fraction: float=0.15, output_file: str='',
            task_column: str='', support_size: int=0, unlabeled_file: str='', unlabeled_id_column: str='', split_column: str='') -> dict:
        """Prepare group-disjoint splits from a measured CSV/TSV or preserve a supplied partition column. Select input and target columns and a biologically justified group column; related observations stay together. IDs may be generated when id_column is empty. Returns evaluation_file, hashes, split counts, a bounded training-only input profile and fit/predict row shapes. With host-protected data, read training_data.file for analysis; evaluation_file is an opaque training/review reference and contains no sample rows. Original values are preserved; no samples are invented or silently discarded. For task=multilabel, target_column contains a JSON object with all label names and 0/1/null values; null is an unobserved label. Optional task_column plus support_size=1..64 creates disjoint training/validation/test tasks with fixed support/query observations; prediction inputs include same-task support. Optional unlabeled_file selects a separate CSV/TSV with the same input columns and no measured-target column; set unlabeled_id_column when supplied. Optional split_column preserves supplied train/validation/test memberships; related groups and connected tasks must remain together. Holdout fractions apply only when split_column is empty; seed still determines episode support selection. Task identity and group annotations must reflect the biological study."""
        from .research_dataset import prepare_dataset
        result=prepare_dataset(self.workspace, data_file=data_file,target_column=target_column,
            input_columns=input_columns,group_column=group_column,provenance=provenance,grouping=grouping,
            id_column=id_column,task=task,seed=seed,validation_fraction=validation_fraction,
            test_fraction=test_fraction,output_file=output_file,data_store=self.data_store,
            task_column=task_column,support_size=support_size,unlabeled_file=unlabeled_file,unlabeled_id_column=unlabeled_id_column,
            split_column=split_column)
        return self.record('prepare_research_dataset',{'source_file':data_file,'source_sha256':result['table_sha256']},result)

    @guard_operation('train_research_model')
    async def train_research_model(self, name: str,
            source_file: Annotated[str, Field(description="Workspace-relative .py file containing complete fit(training, seed) and predict(model, inputs) implementations. This is model code, not the dataset.")],
            evaluation_file: Annotated[str, Field(description="Evaluation reference returned by prepare_research_dataset; pass it unchanged, not as data to pandas or fit. For a revision, use an empty string to reuse the parent experiment's frozen splits.")],
            hypothesis: str, strategy: str='supervised', parent_experiment_id: Annotated[str, Field(description="Use experiment.id from a successfully measured training/evaluation result, never a model name or failed training-attempt ID. Leave empty for the first baseline; a candidate needs a measured parent on the same frozen dataset.")]='',
            variant: ExperimentVariant = "auto", prediction_key: str='value', seed: int=0,
            device: Annotated[Literal["cpu", "cuda"], Field(description="cpu by default. cuda fits in an operator-enabled local PBS GPU allocation; there is no host or CPU fallback. Return CPU-portable weights/parameters: held-out prediction and Agent reuse run on CPU, while independent Review repeats GPU fitting.")]='cpu') -> dict:
        """Fit and validate agent-written model code in the session sandbox. fit(training, seed): rows=training["train"]; each row has an inputs dict and a target. Encode input fields into features before calling a numeric estimator; use the SAME feature transformation in fit and predict. Raw dictionaries or amino-acid strings are not numeric feature vectors. Fit transformations on training rows only and retain them with the learner. Return a parameter dict (arrays/estimators allowed) or fitted estimator. predict(model, inputs) receives ONE inputs dict, not a batch, and returns {"value": prediction}; use prediction_key instead of value if specified. Regression requires a finite number; classification a scalar label; multilabel requires an object mapping every training["labels"] name to a probability in [0,1]. Exclude null target labels during fitting. The host saves fitted state and evaluates held-out data in isolation. Without a parent, evaluation_file creates a baseline. For a revision, cite parent_experiment_id and set evaluation_file="" to reuse frozen splits. Returns measured metrics, fitted files and reproduction entrypoint. Independent Review follows training."""
        from .research_training import fit_model
        args=dict(name=name,source_file=source_file,evaluation_file=evaluation_file,hypothesis=hypothesis,
                  strategy=strategy,parent_experiment_id=parent_experiment_id,variant=variant,
                  prediction_key=prediction_key,seed=seed,device=device)
        progress=(lambda value:write_json(self.control/'execution.json',progress_view(value))) if self.runner is run_python else None
        result=await fit_model(self.workspace,self.development,self.runner,on_progress=progress,**args)
        return self.record('train_research_model',args,result)

    @guard_operation('evaluate_research_model')
    async def evaluate_research_model(self, name: str, source: str, files: list[str], evaluation_file: str, prediction_key: str, hypothesis: str, strategy: str='custom', parent_experiment_id: Annotated[str, Field(description="Use experiment.id from a successfully measured training/evaluation result, never a model name or failed training-attempt ID. Leave empty for the first baseline; a candidate needs a measured parent on the same frozen dataset.")]='', variant: ExperimentVariant = "auto") -> dict:
        """Snapshot a fitted run(inputs) model, replay validation predictions in isolated processes and score on the host. Without a parent, variant=auto starts a baseline and requires an evaluation_file; revised candidates cite parent_experiment_id and set evaluation_file="" to reuse its frozen data/splits. Never use final test scores to choose a generation. Call get_research_strategy_guide for the protocol."""
        args=dict(name=name,source=source,files=files,evaluation_file=evaluation_file,prediction_key=prediction_key,hypothesis=hypothesis,strategy=strategy,parent_experiment_id=parent_experiment_id,variant=variant)
        result=await self.development.evaluate(self.workspace,self.runner,**args)
        return self.record('evaluate_research_model',args,result)

    def workspace_tools(self):
        from langchain_core.tools import StructuredTool

        from mcp_tools.tools.capability_policy import guard_tool
        from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
        from mcp_tools.tools.tool_surface import WORKSPACE_ALIASES
        return [guard_tool(StructuredTool.from_function(coroutine=getattr(self, method), name=name,
                    metadata={'workspace_relative_paths': True}))
                for method, name in WORKSPACE_ALIASES.items() if is_tool_allowed(name)]

    def tools(self):
        from langchain_core.tools import StructuredTool

        from mcp_tools.tools.capability_policy import guard_tool
        from mcp_tools.tools.runtime_tool_policy import is_tool_allowed
        return self.workspace_tools() + [guard_tool(StructuredTool.from_function(coroutine=getattr(self,name),name=name,metadata={"workspace_relative_paths": True, "discovery_category": "training", **({"plan_step_ids": [step["id"] for step in self.plan["steps"]]} if name == "submit_research_checkpoint" else {})})) for name in RESEARCH_TOOLS if is_tool_allowed(name)]

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
    (work/'capability_runner.py').write_text('import importlib.util,json,sys\nspec=importlib.util.spec_from_file_location("candidate",'+repr(manifest['source'])+')\nm=importlib.util.module_from_spec(spec)\nsys.modules[spec.name]=m\nspec.loader.exec_module(m)\ninputs=json.load(open("capability_input.json"))\nresult='+('[m.run(row) for row in inputs]' if batch else 'm.run(inputs)')+'\nwith open("capability_result.json","w") as f: json.dump(result,f,allow_nan=False)\n')
    result=await runner(work,control/'runs','capability_runner.py',120)
    if result['exit_code']!=0:raise ValueError('Capability execution failed: '+result['stderr'][-2000:])
    value=json.loads(read_bytes(work,'capability_result.json',max_bytes=2000000));json.dumps(value,allow_nan=False);return value

def publish_proposals(proposals, goal_id, library=None, *, publication=None):
    from .capability_catalog import register
    from .capability_review import admitted
    if library is None:
        raise ValueError('Choose the private research library; public publication requires operator selection')
    library=Path(library)
    if library.resolve() == LIBRARY.resolve() and publication is None:
        raise ValueError('The platform registry requires operator selection')
    library.mkdir(parents=True,exist_ok=True);published=[]
    for bundle in proposals:
        bundle=Path(bundle);manifest=json.loads((bundle/'manifest.json').read_text());ident=manifest['id']
        if digest({k:v for k,v in manifest.items() if k!='id'})!=ident:raise ValueError('Capability manifest changed')
        receipt=admitted(bundle,manifest)
        if publication is not None:
            from .curated_publication import validate_publication
            validate_publication(publication,manifest,receipt,goal_id)
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
        for name,sha in manifest['files'].items():
            if file_hash(safe_file(dest,name)) != sha: raise ValueError('Stored capability runtime changed')
        if publication is not None: write_json(dest/'publication.json',publication)
        published.append(register(library,manifest,receipt,goal_id,publication=publication))
    return published

def capability_cards(library=LIBRARY):
    from .capability_catalog import cards
    rows = cards(library)
    return [c for c in rows if c.get('visibility') == 'public' and c.get('publication',{}).get('policy') == 'operator_selected.v1'] if Path(library).resolve() == LIBRARY.resolve() else rows

def search_capabilities(query='', limit=10, library=LIBRARY, offset=0):
    from mcp_tools.tools.capability_policy import CapabilityStore

    from .capability_catalog import discovery_card, search
    rows = CapabilityStore().rows()
    prefix = 'model:capability:'
    disabled = [key[len(prefix):] for key,row in rows.items() if key.startswith(prefix) and not row['enabled']]
    page = search(library,query,limit,offset,public_only=Path(library).resolve()==LIBRARY.resolve(),excluded=disabled)
    page['items'] = [discovery_card(card) for card in page['items']]
    page['next_step'] = 'Describe a selected ID for its complete scope, interface and review evidence before running it.'
    return page


def admitted_manifest(capability_id, library):
    from .capability_catalog import require_registered
    from .capability_review import admitted
    if not re.fullmatch('[a-f0-9]{64}',capability_id):raise ValueError('Invalid capability ID')
    bundle=Path(library)/capability_id;manifest=json.loads(safe_file(bundle,'manifest.json').read_text())
    if digest({k:v for k,v in manifest.items() if k!='id'})!=capability_id:raise ValueError('Capability manifest changed')
    receipt=admitted(bundle,manifest);card=require_registered(library,capability_id,receipt['receipt_sha256'])
    if Path(library).resolve() == LIBRARY.resolve():
        from .curated_publication import validate_publication
        if card.get('visibility') != 'public': raise ValueError('Capability has not been published by the operator')
        decision=validate_publication(card.get('publication'),manifest,receipt,card['origin_goal'])
        if json.loads(safe_file(bundle,'publication.json').read_text()) != decision: raise ValueError('Publication record changed')
    for name,sha in manifest['files'].items():
        if file_hash(safe_file(bundle,name))!=sha:raise ValueError('Capability source changed')
    return bundle,manifest,receipt

def describe_capability(capability_id, library=LIBRARY):
    _,manifest,receipt=admitted_manifest(capability_id,library)
    spec=manifest['review_spec']
    from .capability_catalog import require_registered
    card=require_registered(library,capability_id,receipt['receipt_sha256'])
    return {**{k:manifest[k] for k in ['id','name','description','tests','validation']},
            **{k:spec[k] for k in ['kind','author','version','scope','input_schema','output_schema','dependencies']},
            'review_status':'approved','review':receipt,**{k:card[k] for k in ('author_email','visibility','publication') if k in card}}

@guard_operation('run_generated_capability')
async def run_capability(capability_id, inputs, *, library=LIBRARY):
    from .capability_review import schema_validator
    bundle,manifest,_=admitted_manifest(capability_id,library)
    schema_validator(manifest['review_spec']['input_schema']).validate(inputs)
    result=await invoke_bundle(bundle,manifest,inputs,Path(library)/'invocations'/capability_id)
    schema_validator(manifest['review_spec']['output_schema']).validate(result)
    return result
