"""Host-measured model generations and private, reproducible Scientist delivery.

The scientist writes arbitrary algorithms. This module owns scoring, immutable
candidate snapshots, lineage and delivery; model-authored metric claims are not
used as experiment results. Validation feedback is distinct from final Review.
"""
from __future__ import annotations
import hashlib
import ast
import json
import math
from pathlib import Path
import re
import time
import uuid

from .contracts import digest, file_hash
from .host import write_json
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
    return bool(re.search(r'((?:训练|构建|开发|研发|建立|设计|进化|改进|创建|新).{0,40}(?:模型|预测器|分类器|回归器)|(?:模型|预测器).{0,20}(?:研发|训练|迭代|交付)|元学习|半监督|小样本|meta.?learn|semi.?supervis|few.?shot|\b(?:develop|build|train|create|improve|evolve|design)\b.{0,100}\b(?:model|predictor|classifier|regressor)\b)', objective, re.I))


def guide():
    from agent.rsi.artifacts.contract import guide as artifact_guide
    return {'artifact_contract':artifact_guide(), 'strategies':STRATEGIES, 'plm_training_required':False,
        'workflow':['Inspect real data and define leakage-resistant splits',
            'Implement and train a baseline; evaluate_research_model(variant="baseline")',
            'Use returned validation feedback to revise code/learning strategy; evaluate a child with parent_experiment_id',
            'Run relevant ablations and repeat seeds within budget; negative results remain evidence',
            'Freeze a selected generation; propose with review_spec.development_experiment_id',
            'Independent Review checks interfaces, held-out test and retraining; a private model package and test report follow approval'],
        'experiment_api':{'source':'Python module implementing run(inputs: dict) -> dict',
            'files':'Runtime dependencies and fitted weights only, no raw datasets',
            'evaluation_file':'JSON matching get_research_review_requirements; claimed_metric is used only by final Review',
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


def evaluation_digest(data):
    # A final metric claim is checked by Review, never part of the data identity.
    return digest({k:v for k,v in data.items() if k!='claimed_metric'})


def validate_data(data, strategy):
    require(data.get('task') in {'regression','classification'},'Declare regression or classification')
    require(all(isinstance(data.get(k),str) and data[k].strip() for k in ['provenance','grouping']),'Declare data provenance and grouping')
    seen_ids=set();seen_groups=set();seen_inputs=set();task_sets=[]
    for name,minimum in [('train',5),('validation',2),('test',3)]:
        rows=data.get('splits',{}).get(name)
        require(isinstance(rows,list) and minimum<=len(rows)<=(20000 if name=='train' else 512),'Provide bounded train/validation/test splits')
        ids=set();groups=set();inputs=set();tasks=set()
        for row in rows:
            require(isinstance(row,dict) and all(isinstance(row.get(k),str) and row[k] for k in ['id','group']) and isinstance(row.get('inputs'),dict),'Each row needs id, group and inputs')
            target=row.get('target');require(type(target) in (int,float,str) and (isinstance(target,str) or math.isfinite(target)),'Provide finite targets')
            if data['task']=='regression':require(type(target) in (int,float),'Regression requires numeric targets')
            require(row['id'] not in ids,'Duplicate sample ID');ids.add(row['id']);groups.add(row['group']);inputs.add(digest(row['inputs']))
            if strategy=='meta_learning':
                require(isinstance(row.get('task_id'),str) and row['task_id'],'Meta-learning requires task_id for every row');tasks.add(row['task_id'])
        require(not(ids&seen_ids or groups&seen_groups or inputs&seen_inputs),'Leakage: IDs, groups or inputs overlap splits')
        seen_ids|=ids;seen_groups|=groups;seen_inputs|=inputs;task_sets.append(tasks)
    if strategy=='meta_learning':
        require(len(task_sets[0])>=2 and all(task_sets[1:]) and not any(task_sets[i]&task_sets[j] for i in range(3) for j in range(i)),'Meta-learning requires multiple training tasks and disjoint validation/test tasks')
    if strategy=='semi_supervised':
        rows=data.get('unlabeled',[]);require(isinstance(rows,list) and 1<=len(rows)<=20000,'Semi-supervised learning requires genuine unlabeled observations')
        for row in rows:
            require(isinstance(row,dict) and set(row)=={'id','inputs'} and isinstance(row['id'],str) and row['id'] and isinstance(row['inputs'],dict),'Unlabeled rows contain id and inputs only')
            key=digest(row['inputs']);require(row['id'] not in seen_ids and key not in seen_inputs,'Unlabeled data overlap labeled or held-out observations')
            seen_ids.add(row['id']);seen_inputs.add(key)
    json.dumps(data,allow_nan=False)
    return data


def scores(task, predictions, targets, train):
    require(len(predictions)==len(targets) and bool(targets),'Prediction count differs from query count')
    if task=='regression':
        require(all(type(p) in (int,float) and math.isfinite(p) for p in predictions),'Predictions must be finite numbers')
        mean=sum(train)/len(train)
        return {'metric':'mae','direction':'minimize','score':sum(abs(p-y) for p,y in zip(predictions,targets))/len(targets),
            'rmse':math.sqrt(sum((p-y)**2 for p,y in zip(predictions,targets))/len(targets)),
            'reference':sum(abs(mean-y) for y in targets)/len(targets),'reference_method':'training_mean'}
    from collections import Counter
    require(all(type(p) in (int,float,str) and (isinstance(p,str) or math.isfinite(p)) for p in predictions),'Predictions must be finite scalar class labels')
    majority=Counter(train).most_common(1)[0][0]
    return {'metric':'accuracy','direction':'maximize','score':sum(p==y for p,y in zip(predictions,targets))/len(targets),
        'reference':sum(majority==y for y in targets)/len(targets),'reference_method':'training_majority'}


class Development:
    def __init__(self, root):
        self.root=Path(root)

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

    async def evaluate(self, workspace, runner, *, name, source, files, evaluation_file, prediction_key, hypothesis, strategy, parent_experiment_id='', variant='candidate'):
        from .research_workspace import invoke_bundle
        require(strategy in STRATEGIES,'Use a declared strategy or custom')
        require(variant in {'baseline','candidate','ablation'},'Invalid experiment variant')
        require(isinstance(name,str) and 0<len(name.strip())<=120 and isinstance(hypothesis,str) and 0<len(hypothesis.strip())<=4000,'Describe the model and testable hypothesis')
        require(isinstance(files,list) and len(files)<=20 and all(isinstance(p,str) for p in files),'Provide at most 20 runtime files')
        require(isinstance(prediction_key,str) and 0<len(prediction_key)<=100,'Declare prediction_key')
        require(len(list((self.root/'experiments').glob('*')))<64,'Research experiment budget reached')
        raw=read_bytes(workspace,evaluation_file,max_bytes=20*1024**2);data=validate_data(json.loads(raw),strategy)
        data_sha=evaluation_digest(data)
        parent=self.record(parent_experiment_id) if parent_experiment_id else None
        if parent:require(parent['evaluation_sha256']==data_sha,'Compare generations on identical data and splits')
        require(parent is not None or variant=='baseline','Start with a measured baseline; later generations need a parent_experiment_id')
        names=sorted(set([source,*files]));require(evaluation_file not in names,'Evaluation data must not be a runtime file')
        require(not set(Path(n).name for n in names)&{'manifest.json','admission.json','capability_input.json','capability_result.json','capability_runner.py'},'Reserved runtime filename')
        identity=uuid.uuid4().hex;folder=self.root/'experiments'/identity;bundle=folder/'runtime';bundle.mkdir(parents=True)
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
        comparison=None
        if parent:
            before=parent['validation'];delta=before['score']-measured['score'] if measured['direction']=='minimize' else measured['score']-before['score']
            comparison={'parent_score':before['score'],'delta_in_preferred_direction':delta,'improved_on_validation':delta>0,'statistical_significance':'not_established'}
        record={'id':identity,'created':time.time(),'name':name,'strategy':strategy,'variant':variant,'hypothesis':hypothesis,'parent':parent_experiment_id or None,
            'source':source,'files':manifest['files'],'runtime_sha256':fingerprint,'implementation_sha256':implementation,'evaluation_sha256':data_sha,'evaluation_file':evaluation_file,
            'prediction_key':prediction_key,'split_counts':{k:len(v) for k,v in data['splits'].items()},'validation':measured,'comparison':comparison,
            'predictions_sha256':digest(results),'fresh_process_replay':True,'score_origin':'host_validation','test_used_for_feedback':False}
        record['receipt_sha256']=digest(record);write_json(folder/'receipt.json',record)
        # User-facing evidence is regenerated from the host receipt, never read
        # back as authority when a candidate is proposed or registered.
        write_bytes(workspace,'experiments/'+identity+'.json',json.dumps(record,indent=2,ensure_ascii=False).encode())
        return record

    def selected(self, identity, manifest, data):
        row=self.record(identity)
        require(row['source']==manifest['source'] and row['files']==manifest['files'],'Selected runtime differs from the measured model generation')
        require(row['evaluation_sha256']==evaluation_digest(data),'Evaluation data changed after the recorded experiment')
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
