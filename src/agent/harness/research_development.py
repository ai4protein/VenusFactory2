"""Host-measured model generations and private, reproducible Scientist delivery.

The scientist writes arbitrary algorithms. This module owns scoring, immutable
candidate snapshots, lineage and delivery; model-authored metric claims are not
used as experiment results. Validation feedback is distinct from final Review.
"""
from __future__ import annotations
import hashlib
import ast
import json
from pathlib import Path
import re
import time
import uuid

from .contracts import digest, file_hash
from .host import write_json
from .research_prediction_diagnostics import prediction_diagnostics
from .research_evaluation import evaluation_digest, validate_data, scores
from utils.scoped_files import read_bytes, copy_file, write_bytes

STRATEGIES = {
    'supervised': {'use':'Supervised baselines and new custom architectures; linear/kernel/tree/neural methods are all valid.',
        'comparison':'Training-mean/majority reference and a credible task-specific baseline.'},
    'few_shot': {'use':'Few labeled observations per target; regularized adaptation, kernels, prototypes or learned priors.',
        'comparison':'Same support examples and query split; several support sizes/seeds; uncertainty and failures.'},
    'semi_supervised': {'use':'Labeled plus genuine unlabeled observations from a documented compatible distribution.',
        'comparison':'Labeled-only ablation on exactly the same validation query records; report pseudo-label filtering and calibration.'},
    'meta_learning': {'use':'Several training tasks and disjoint validation/test tasks; learned initialization, metric or update rule.',
        'comparison':'No-adaptation and ordinary fine-tuning; fixed support/query episodes and adaptation budget; several seeds.'},
    'transfer_learning': {'use':'Transfer across proteins, families or assays with a documented source and domain shift.',
        'comparison':'Target-only learner and no-transfer ablation; separate source/target groups.'},
    'custom': {'use':'A new hypothesis-driven algorithm or combination of strategies implemented by the scientist.',
        'comparison':'Declare mechanism, matched baseline, ablation, cost and scope; novelty needs a literature comparison.'},
}


def model_goal(objective):
    # Reproduction and delivery still produce a reusable model. They require the
    # same measured evidence and independent admission as newly developed models.
    return bool(re.search(
        r'(?:(?:训练|拟合|构建|开发|研发|建立|设计|进化|改进|创建|复现|复训|交付|新)'
        r'.{0,40}(?:模型|预测器|分类器|回归器)|'
        r'(?:模型|预测器).{0,20}(?:研发|训练|迭代|复现|复训|交付)|'
        r'元学习|半监督|小样本|meta.?learn|semi.?supervis|few.?shot|'
        r'\b(?:develop|build|train|retrain|fit|create|improve|evolve|design|'
        r'reproduce|replicate|implement|deliver)\b.{0,120}'
        r'\b(?:model|predictor|classifier|regressor|regression[ -]project)\b)',
        objective, re.I | re.S))


def guide():
    from agent.rsi.artifacts.contract import guide as artifact_guide
    return {'artifact_contract':artifact_guide(), 'strategies':STRATEGIES, 'plm_training_required':False,
        'workflow':['Inspect real data and define leakage-resistant splits',
            'Use prepare_research_dataset for measured CSV/TSV data, or write a custom JSON and validate_research_evaluation',
            'Write fit(training, seed) and predict(model, inputs); train_research_model(variant="baseline") fits and measures them',
            'Use returned validation feedback to revise code/learning strategy; evaluate a child with parent_experiment_id',
            'Run relevant ablations and repeat seeds within budget; negative results remain evidence',
            'Freeze a selected generation; propose with review_spec.development_experiment_id',
            'Independent Review checks interfaces, held-out test and retraining; a private model package and test report follow approval'],
        'training_api':{'tool':'train_research_model',
            'source_file':'A saved Python module: fit(training, seed) -> fitted parameter dict (arrays allowed) or estimator; predict(model, inputs) -> output dict.',
            'training':'training["train"] is a list: X=[row["inputs"] for row in training["train"]]; y=[row["target"] for row in training["train"]]. Other keys: task, strategy and genuine unlabeled rows. No validation/test labels enter fitting.',
            'outputs':'Fitted runtime and model state, reproducible training script/data, validation metrics and experiment ID. Use evaluate_research_model for an already-fitted custom runtime.'},
        'experiment_api':{'source':'Python module implementing run(inputs: dict) -> dict',
            'files':'Runtime dependencies and fitted weights only, no raw datasets',
            'evaluation_file':'Baseline: a JSON matching get_research_review_requirements. Child: set an empty string to reuse the parent frozen evaluation; never edit evaluation targets. claimed_metric is used only by final Review',
            'restore_research_evaluation':'Export an unchanged frozen evaluation to a new workspace filename for final Review; existing different files are never overwritten',
            'task_id':'For meta_learning each split row needs a task_id; train, validation and test tasks must be disjoint. Put only legitimate support observations in each query input, never its own target.',
            'unlabeled':'For semi_supervised add an unlabeled list of {id, inputs}; no targets, no overlap with any labeled split.',
            'strategy':'Choose a strategy ID or custom; justify the actual algorithm in hypothesis',
            'parent_experiment_id':'Previously measured generation with the identical evaluation-data hash',
            'variant':'baseline, candidate or ablation'},
        'evidence':'Metrics are computed by the host on validation queries. Test results belong to final admission; supplied splits are developer-visible, not a concealed benchmark. Reports must state this limitation. No mandatory superiority threshold.',
        'delivery':'Fitted runtime, training source, input/output contract, model card, validation lineage, independent test report, examples, dependency versions and checksums; raw evidence stays private.',
        'online':'CPU sandbox budgets and administrator policy apply; no PLM/GPU requirement and no paid LLM fallback.',
        'references':['https://proceedings.mlr.press/v70/finn17a.html','https://scikit-learn.org/stable/modules/semi_supervised.html']}


def require(value, message):
    if not value: raise ValueError(message)


def inspect_evaluation(workspace, evaluation_file, strategy='supervised', *, data_store=None):
    """Check the same partition contract used by measured generations."""
    require(strategy in STRATEGIES, 'Use a declared strategy or custom')
    raw=data_store.load_evaluation(evaluation_file) if data_store else read_bytes(workspace,evaluation_file,max_bytes=20*1024**2)
    data=validate_data(json.loads(raw),strategy)
    splits={}
    for name,rows in data['splits'].items():
        if name not in {'train','validation','test'}: continue
        splits[name]={
            'rows':len(rows),
            'groups':len({row['group'] for row in rows}),
            'unique_inputs':len({digest(row['inputs']) for row in rows}),
        }
    return {
        'passed':True, 'validation_scope':'partition_schema_and_disjointness',
        'evaluation_file':evaluation_file,
        'source_sha256':hashlib.sha256(raw).hexdigest(),
        'evaluation_sha256':evaluation_digest(data),
        'task':data['task'], 'strategy':strategy, 'splits':splits,
        'unlabeled_rows':len(data.get('unlabeled',[])),
        'checks':['row_limits','finite_targets','disjoint_ids','disjoint_groups','disjoint_exact_inputs'],
    }


def immutable_copy(root, path, raw):
    """Create an anchored file once; an existing different file is never replaced."""
    from utils.scoped_files import parent_fd
    import os
    with parent_fd(root,path,create=True) as (directory,name):
        try:
            fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=directory)
        except FileExistsError:
            require(read_bytes(root,path,max_bytes=20*1024**2)==raw,
                    'Output already contains different data; choose a new output_file')
        else:
            with os.fdopen(fd,'wb') as stream:
                stream.write(raw)


class Development:
    def __init__(self, root, *, data_store=None):
        self.data_store=data_store
        self.root=Path(root)

    def load_evaluation(self, workspace, path):
        if self.data_store is not None:
            return self.data_store.load_evaluation(path)
        return read_bytes(workspace,path,max_bytes=20*1024**2)

    def write_evaluation(self, workspace, path, raw):
        if self.data_store is not None:
            return self.data_store.write_evaluation(path,raw)
        immutable_copy(workspace,path,raw)

    def records(self):
        return sorted([json.loads(p.read_text()) for p in self.root.glob('experiments/*/receipt.json')],key=lambda r:r['created'])

    def record(self, identity):
        require(isinstance(identity,str) and re.fullmatch('[a-f0-9]{32}',identity),'Invalid experiment ID')
        path=self.root/'experiments'/identity/'receipt.json'
        require(path.is_file(),'Unknown experiment ID')
        row=json.loads(path.read_text())
        require(row['receipt_sha256']==digest({k:v for k,v in row.items() if k!='receipt_sha256'}),'Experiment receipt changed')
        return row

    def summary(self):
        rows=self.records()
        return {'count':len(rows),'generations':[{k:r[k] for k in ['id','name','strategy','variant','parent','hypothesis','validation','comparison','runtime_sha256']} for r in rows[-12:]]}

    def evaluation_snapshot(self, identity):
        record=self.record(identity)
        if self.data_store is not None:
            require(record.get('evaluation_visibility')=='host_only','This prior evaluation was agent-visible; start a fresh protected evaluation')
        expected=record.get('evaluation_snapshot_sha256')
        require(expected,
            'This historic experiment has no frozen evaluation snapshot. Supply an evaluation_file '
            'matching its original evaluation hash; never reconstruct or alter labels.')
        snapshot=record.get('evaluation_snapshot_id')
        require(isinstance(snapshot,str) and re.fullmatch('[a-f0-9]{64}',snapshot),'Invalid frozen evaluation reference')
        raw=read_bytes(self.root/'evaluations',snapshot+'.json',max_bytes=20*1024**2)
        require(hashlib.sha256(raw).hexdigest()==expected,'Frozen evaluation snapshot changed')
        data=json.loads(raw)
        require(evaluation_digest(data)==record['evaluation_sha256'],'Frozen evaluation identity changed')
        return record,raw,data

    def export_evaluation(self, workspace, identity, output_file=''):
        record,raw,data=self.evaluation_snapshot(identity)
        path=output_file or 'evaluations/'+record['evaluation_sha256']+'.json'
        self.write_evaluation(workspace,path,raw)
        return {'path':path,'experiment_id':identity,'evaluation_sha256':record['evaluation_sha256'],
                'sha256':hashlib.sha256(read_bytes(workspace,path,max_bytes=20*1024**2)).hexdigest(),'split_counts':record['split_counts'],
                'data_access':'host_only' if self.data_store else 'workspace',
                'next_step':'Use this unchanged evaluation file as private evidence for final Review; '
                'training-only preprocessing belongs in model code, not in evaluation targets. '
                'The optional claimed_metric may be omitted; host metrics are computed independently.'}

    async def evaluate(self, workspace, runner, *, name, source, files, evaluation_file, prediction_key, hypothesis, strategy, parent_experiment_id='', variant='auto'):
        from .research_workspace import invoke_bundle
        require(strategy in STRATEGIES,'Use a declared strategy or custom')
        if variant == 'auto':
            variant = 'candidate' if parent_experiment_id else 'baseline'
        require(variant in {'baseline','candidate','ablation'},'Invalid experiment variant')
        require(isinstance(name,str) and 0<len(name.strip())<=120 and isinstance(hypothesis,str) and 0<len(hypothesis.strip())<=4000,'Describe the model and testable hypothesis')
        require(isinstance(files,list) and len(files)<=20 and all(isinstance(p,str) for p in files),'Provide at most 20 runtime files')
        require(isinstance(prediction_key,str) and 0<len(prediction_key)<=100,'Declare prediction_key')
        require(len(list((self.root/'experiments').glob('*')))<64,'Research experiment budget reached')
        parent=self.record(parent_experiment_id) if parent_experiment_id else None
        require(isinstance(evaluation_file,str),'evaluation_file must be a workspace filename or empty for parent reuse')
        if not evaluation_file:
            require(parent is not None,'The first baseline requires an evaluation_file')
            _,raw,data=self.evaluation_snapshot(parent_experiment_id)
            evaluation_file=parent['evaluation_file']
            evaluation_origin={'kind':'parent_snapshot','experiment_id':parent_experiment_id}
        else:
            raw=self.load_evaluation(workspace,evaluation_file)
            data=json.loads(raw)
            evaluation_origin={'kind':'workspace_file','path':evaluation_file}
        data=validate_data(data,strategy)
        data_sha=evaluation_digest(data)
        if parent:require(parent['evaluation_sha256']==data_sha,
            'Compare generations on identical data and splits. Keep evaluation targets and group assignments unchanged. '
            'Set evaluation_file="" to reuse the parent snapshot; call restore_research_evaluation for a fresh evidence copy. '
            'Training-only transformations belong in the model/training code.')
        require(parent is not None or variant=='baseline','Start with a measured baseline; later generations need a parent_experiment_id')
        names=sorted(set([source,*files]));require(evaluation_file not in names,'Evaluation data must not be a runtime file')
        require(not set(Path(n).name for n in names)&{'manifest.json','admission.json','capability_input.json','capability_result.json','capability_runner.py'},'Reserved runtime filename')
        identity=uuid.uuid4().hex;folder=self.root/'experiments'/identity;bundle=folder/'runtime';bundle.mkdir(parents=True)
        snapshot_raw=json.dumps({k:v for k,v in data.items() if k!='claimed_metric'},
                                sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
        evaluation_root=self.root/'evaluations';evaluation_root.mkdir(parents=True,exist_ok=True)
        immutable_copy(evaluation_root,data_sha+'.json',snapshot_raw)
        for path in names:copy_file(workspace,path,bundle,path)
        require(sum((bundle/p).stat().st_size for p in names)<=200*1024**2,'A CPU experiment runtime is limited to 200 MB')
        manifest={'source':source,'files':{p:file_hash(bundle/p) for p in names}}
        fingerprint=digest(manifest)
        implementation=digest({p:(ast.dump(ast.parse(read_bytes(bundle,p).decode()),include_attributes=False) if p.endswith('.py') else manifest['files'][p]) for p in names})
        if parent:require(parent.get('implementation_sha256',parent['runtime_sha256'])!=implementation,'A new generation must change its implementation or fitted artifacts, not only comments')
        queries=[r['inputs'] for r in data['splits']['validation']]
        results=await invoke_bundle(bundle,manifest,queries,folder/'inference',runner=runner,batch=True)
        replay=await invoke_bundle(bundle,manifest,queries,folder/'replay',runner=runner,batch=True)
        require(digest(results)==digest(replay),'Predictions changed on fresh-process replay')
        require(isinstance(results,list) and all(isinstance(v,dict) and prediction_key in v for v in results),'Return the declared prediction_key for each query')
        measured=scores(data['task'],[v[prediction_key] for v in results],[r['target'] for r in data['splits']['validation']],[r['target'] for r in data['splits']['train']])
        measured['diagnostics']=prediction_diagnostics(data['task'],[v[prediction_key] for v in results],
            [r['target'] for r in data['splits']['train']],split='validation')
        comparison=None
        if parent:
            before=parent['validation'];delta=before['score']-measured['score'] if measured['direction']=='minimize' else measured['score']-before['score']
            comparison={'parent_score':before['score'],'delta_in_preferred_direction':delta,'improved_on_validation':delta>0,'statistical_significance':'not_established'}
        record={'id':identity,'created':time.time(),'name':name,'strategy':strategy,'variant':variant,'hypothesis':hypothesis,'parent':parent_experiment_id or None,
            'source':source,'files':manifest['files'],'runtime_sha256':fingerprint,'implementation_sha256':implementation,'evaluation_sha256':data_sha,'evaluation_file':evaluation_file,
            'evaluation_snapshot_id':data_sha,'evaluation_snapshot_sha256':hashlib.sha256(snapshot_raw).hexdigest(),'evaluation_origin':evaluation_origin,
            'prediction_key':prediction_key,'split_counts':{k:len(v) for k,v in data['splits'].items()},'validation':measured,'comparison':comparison,
            'predictions_sha256':digest(results),'fresh_process_replay':True,'score_origin':'host_validation','test_used_for_feedback':False}
        record['evaluation_visibility']='host_only' if self.data_store else 'agent_workspace'
        record['receipt_sha256']=digest(record);write_json(folder/'receipt.json',record)
        # User-facing evidence is regenerated from the host receipt, never read
        # back as authority when a candidate is proposed or registered.
        write_bytes(workspace,'experiments/'+identity+'.json',json.dumps(record,indent=2,ensure_ascii=False).encode())
        return record

    def selected(self, identity, manifest, data):
        row=self.record(identity)
        if row['source'] != manifest['source'] or row['files'] != manifest['files']:
            changed = [name for name, sha in row['files'].items()
                       if name in manifest['files'] and manifest['files'][name] != sha]
            raise ValueError(
                'Selected runtime differs from the measured model generation. '
                'Expected source: ' + json.dumps(row['source']) +
                '; expected runtime files: ' + json.dumps(list(row['files'])) +
                '; changed files: ' + json.dumps(changed) +
                '. Use the same measured source and files, or evaluate the changed runtime '
                'as a new generation before proposing it. See experiments/' + identity + '.json.')
        require(row['evaluation_sha256']==evaluation_digest(data),'Evaluation data changed after the recorded experiment. Use restore_research_evaluation(experiment_id) and include the returned unchanged target/split data as private Review evidence.')
        chain=[row];seen={row['id']}
        while chain[-1]['parent']:
            parent=self.record(chain[-1]['parent']);require(parent['id'] not in seen,'Cyclic model lineage')
            require(parent['evaluation_sha256']==row['evaluation_sha256'],'Incomparable lineage data')
            seen.add(parent['id']);chain.append(parent)
        # Research may legitimately retain the baseline after a failed revision.
        # Require a measured descendant, not a mandatory accuracy win or a forced
        # deployment of the more complicated model.
        if len(chain)==1 and row['variant']=='baseline':
            for candidate in reversed(self.records()):
                if candidate['evaluation_sha256']!=row['evaluation_sha256']:continue
                branch=[self.record(candidate['id'])];visited={candidate['id']}
                while branch[-1]['parent'] and branch[-1]['id']!=identity:
                    parent=self.record(branch[-1]['parent']);require(parent['id'] not in visited,'Cyclic model lineage')
                    visited.add(parent['id']);branch.append(parent)
                if len(branch)>1 and branch[-1]['id']==identity:
                    chain=branch;break
        require(len(chain)>=2 and chain[-1]['variant']=='baseline','Model delivery requires a measured baseline and a revised generation')
        return {'selected_experiment_id':identity,'generations':list(reversed(chain)),'claim':'Measured development trajectory; biological novelty and superiority are not inferred from the strategy name.'}


# Public import retained for saved clients and existing development integrations.
from agent.rsi.artifacts.package import build_delivery
