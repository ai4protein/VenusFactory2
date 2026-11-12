"""Natural-language Scientist goals: plan, execute, review and continue."""
from __future__ import annotations
import asyncio
import copy
from dataclasses import replace
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time

import httpx

from agent.science_goal import GoalSpec, GoalStore, run_goal
from .scientist_gateway_policy import ScientistGatewayPolicy, ScientistResponsesPolicy
from .contracts import Profile, file_hash, digest
from .events import audit_requests
from .research_execution_feedback import execution_feedback, iteration_feedback, feedback_state, repair_source_context, failure_summary
from .gateway import EvidenceGateway
from .host import write_json
from .native_expert_team import NATIVE_EXPERT_HARNESSES, critic_decision, role_content
from .research_updates import apply_revisions
from .research_plan_evidence import VERIFICATION_KINDS, checkpoint_steps
from .research_focus import selected_tools, instructions as execution_instructions
from .research_questions import validate_input_questions
from .research_workspace import ResearchWorkspace, publish_proposals, safe_file

from .capability_review import REVIEW_PROMPT, review_proposal
from .research_development import Development, model_goal, build_delivery
from .research_topic import SCIENTIST_PURPOSE, topic_context
from . import scientific_acceptance as sc_acceptance
from .tool_disclosure import ToolDisclosure, CONTROLS, GUIDANCE
from agent.rsi.artifacts.contract import SCHEMA as DELIVERY_SCHEMA, selected_deliveries, has_delivery

def development_next_action(summary, progress, prepared=None, *, materials=None):
    action = _development_next_action(summary, progress, prepared, materials=materials)
    feedback = progress.get('sc_revision') or {}
    if feedback.get('issues') and not feedback.get('requests'):
        # Study-level feedback still reaches PI when no model was submitted.
        # Keep the measured workflow stage: a missing proposal is not a failed fit.
        action['scientific_feedback'] = {
            'summary': feedback.get('summary', ''), 'issues': feedback['issues'],
            'measured_experiment_ids': feedback.get('measured_experiment_ids', []),
            'instruction': 'Address SC study feedback using the existing host measurements. '
                'Distinguish a missing experiment from incomplete submission or delivery; '
                'preserve completed evidence and revise only the unresolved work.'}
    return action


def _development_next_action(summary, progress, prepared=None, *, materials=None):
    """Expose only the next missing model-development stage."""
    sc_revision = progress.get('sc_revision')
    if sc_revision and sc_revision.get('requests'):
        return {'stage':'revise_scientific_model', 'tool':'train_research_model',
                'issues':sc_revision['issues'], 'requests':sc_revision.get('requests', []),
                'rejected_experiment_ids':sc_revision.get('experiment_ids', []),
                'instruction':'SC did not accept the model usefulness. PI must use this evidence to revise the scientific approach, comparison or supported scope, then implement and measure the required experiment. Do not repeat unchanged work or claim success from execution alone. Preserve the objective and acceptance criteria. Submit a supported candidate for a fresh SC assessment and independent technical Review.'}
    revisions = [r for r in progress.get('publication_reviews', []) if r.get('status') != 'approved']
    if revisions:
        return {'stage':'revise_submission', 'tool':'propose_research_capability',
                'proposal_file':progress.get('review_submission',{}).get('proposal_file'),
                'issues':[str(issue)[:500] for row in revisions[-2:] for issue in row.get('issues', [])[:3]],
                'instruction':'Edit the saved proposal file to fix the reported issues, preserving its unchanged author, scope and interface fields. Resubmit with proposal_file. Submission remains subject to automatic verification and independent Review.'}
    if has_delivery(progress.get('deliveries', []), 'model'):
        return {'stage':'finish_research', 'instruction':'Use the reviewed delivery evidence to finish the report and checkpoint the remaining plan steps.'}
    generations = summary.get('generations', [])
    candidates = [row['id'] for row in generations if row.get('variant') == 'candidate' and row.get('parent')]
    if candidates:
        return {'stage':'submit_for_review', 'tool':'propose_research_capability',
                'eligible_experiment_ids':candidates[-8:],
                'instruction':'Evaluation records an experiment; it does not submit independent review. Select using matched validation evidence. Inspect experiments/<id>.json for the measured source and runtime file list; reuse them unchanged or evaluate a changed runtime first. Inspect the proposal interface and review requirements, then call propose_research_capability with review_spec.development_experiment_id. Do this even if all scientific plan steps are marked complete.'}
    baselines = [row['id'] for row in generations if row.get('variant') == 'baseline']
    if baselines:
        return {'stage':'evaluate_revision', 'tool':'train_research_model',
                'eligible_parent_ids':baselines[-8:], 'evaluation_tool':'evaluate_research_model',
                'instruction':'Inspect the baseline validation diagnostics. If predictions are constant or match the training reference, check feature variation and learned parameters on training data. Implement and fit a justified revision, then evaluate variant=candidate with the measured baseline parent_experiment_id and identical data splits. A constant baseline may be retained when its purpose and measured limitations are explicit.'}
    action = {'stage':'evaluate_baseline', 'tool':'prepare_research_dataset',
              'evaluation_tool':'evaluate_research_model'}
    if prepared is not None:
        # Preparing observations does not establish an executable model. Point
        # at verified supplied source before suggesting another implementation.
        sources = [{'path':row['path'], 'sha256':row['sha256']}
                   for row in (materials or {}).get('files', [])
                   if isinstance(row,dict) and isinstance(row.get('path'),str)
                   and row['path'].endswith('.py') and row.get('sha256')][:4]
        action['evaluation_file'] = prepared['evaluation_file']
        action['tool'] = 'read_session_file' if sources else 'write_session_file'
        if sources:
            action['supplied_sources'] = sources
            action['instruction'] = ('Read the supplied Python source with read_session_file before writing a replacement. '
                'Check its task, features and fit/predict contract; reuse compatible code and preserve its attribution. ')
        else:
            action['instruction'] = 'Implement a complete fit(training, seed) and predict(model, inputs) module before fitting. '
        action['instruction'] += ('Data preparation is complete, but no baseline has been measured. '
            'Reuse this evaluation_file; the visible table contains training rows only. Do not prepare or repartition it again. ')
    else:
        action['dataset_tool'] = 'prepare_research_dataset'
        action['instruction'] = 'Inspect table headers and prepare_research_dataset by biological groups. '
    action['instruction'] += ('Sequence regression and multilabel studies can inspect get_research_baseline for a compatible fixed control. '
        'Do not overwrite supplied code with placeholders. Fit the complete saved source with train_research_model(variant=baseline); '
        'use the actual source path, not an invented filename. Already-fitted runtimes can use evaluate_research_model.')
    return action


DRIVER='autonomous_research.v1'
CRITERIA=('research_plan_executed','research_outputs_reviewed','capability_admission_resolved')
ROLES={
 'REVIEW':REVIEW_PROMPT,
 'PI':"You are the Principal Investigator. Interpret the user's protein, property, materials and research objective. Choose one testable next experiment using the available data, tools and budget. For model development, propose a plausible feature representation, learning method and conventional comparison, and explain the scientific reason for the next revision using measured feedback. State assumptions that data inspection must check. Preserve the actual target and observed measurements. Return a concise public research direction; execution results come from the implementation phase.",
 'CB':'You are the Computational Biologist. Turn the PI direction into 3..6 feasible milestones (maximum 24) for this objective and its supplied materials. Return one JSON object: {"summary":"...","deliverable":"model","steps":[{"id":"step1","title":"...","acceptance":"measured endpoint and evidence","approach":"specific method and comparison","verification":"model"}],"needs_input":[]}. For each training or revision step, name the feature representation, learner or training procedure, comparison and scientific rationale in approach. Choose a conventional baseline suited to the data; get_research_baseline provides sequence regression and multilabel controls when applicable, with reference provenance. Use enabled tools and inspect actual inputs before setting sample counts or performance requirements. Acceptance fixes the research endpoint, observed data, splits and required evidence; approach records the method and may change after failed experiments. Preserve existing step IDs, acceptance criteria and verified completed steps. The host manages status and evidence. Use progress.user_updates and revision_history when replanning. Place needs_input only at the plan top level, for specific missing information that blocks every useful next step and cannot be obtained from supplied materials or public sources. Choose reasonable assumptions for optional preferences. Seek public measured data through enabled tools when necessary; never replace missing measurements with invented data or a different property\'s proxy labels. Describe prospective work; do not claim experiments or results before execution.',
 'SC':sc_acceptance.PROMPT,
}

def parse_plan(text, previous=None, *, objective=''):
    text=text.strip()
    if text.startswith('```') and text.endswith('```'):text='\n'.join(text.splitlines()[1:-1])
    try:
        value=json.loads(text)
    except json.JSONDecodeError:
        # Some free models wrap their JSON in a short introduction or a fence.
        # Accept exactly one complete plan; never infer missing fields or steps.
        decoder=json.JSONDecoder();plans=[]
        for index, character in enumerate(text):
            if character!='{':continue
            try: candidate,_=decoder.raw_decode(text[index:])
            except json.JSONDecodeError:continue
            if isinstance(candidate,dict) and {'summary','steps','needs_input'}<=set(candidate) and not set(candidate)-{'summary','steps','needs_input','deliverable'}:
                plans.append(candidate)
        if len(plans)!=1:raise ValueError('Return one complete research plan JSON object')
        value=plans[0]
    if not isinstance(value,dict) or set(value)-{'summary','steps','needs_input','deliverable'} or not {'summary','steps','needs_input'}<=set(value) or not isinstance(value['summary'],str) or not isinstance(value['steps'],list) or not 1<=len(value['steps'])<=24 or not isinstance(value['needs_input'],list) or any(not isinstance(x,str) for x in value['needs_input']):raise ValueError('Invalid research plan')
    validate_input_questions(value['needs_input'])
    if 'deliverable' in value and value['deliverable'] not in {'analysis','tool','model'}:raise ValueError('Invalid research deliverable')
    ids=set()
    for index, row in enumerate(value['steps']):
        required={'id','title','acceptance'}
        if isinstance(row,dict) and set(row)-required-{'status','evidence','verification','approach'}:
            extra=', '.join(sorted(set(row)-required-{'status','evidence','verification','approach'}))[:200]
            raise ValueError(f'Step {index+1} has unsupported fields: {extra}. '
                'Each step contains id, title, acceptance and optional verification and approach fields. Put needs_input at the plan top level, not inside steps. '
                'Do not supply completion status or evidence; the host owns those fields.')
        if not isinstance(row,dict) or not required<=set(row) or set(row)-required-{'status','evidence','verification','approach'} or any(not isinstance(row[k],str) or not row[k].strip() or len(row[k])>2000 for k in required) or row['id'] in ids:
            raise ValueError('Each research step requires unique id, title and acceptance strings; status and evidence are managed by the host')
        if 'approach' in row and (not isinstance(row['approach'],str) or not row['approach'].strip() or len(row['approach'])>2000):
            raise ValueError('A research approach must be a concise method description')
        if row.get('verification','artifact') not in VERIFICATION_KINDS:
            raise ValueError('Unsupported plan verification kind')
        # A planner may echo its input, including checkpoint metadata. Never
        # trust that metadata: restore only previously verified host state.
        ids.add(row['id']);row['status']='pending';row['evidence']=[]
    old={r['id']:r for r in (previous or {}).get('steps',[])}
    missing=set(old)-ids
    if missing:
        # Models often return only the remaining steps. Preserve the original
        # requirements on the host instead of spending a whole cycle copying
        # completed rows. An unrelated replacement plan is still rejected.
        if not (set(old)&ids) and any(r.get('status')!='completed' for r in old.values()):
            raise ValueError('A revised plan cannot silently drop existing requirements')
        value['steps']=[copy.deepcopy(r) for r in old.values() if r['id'] in missing]+value['steps']
        if len(value['steps'])>24:raise ValueError('A revised plan exceeds the step limit')
    for row in value['steps']:
        if row['id'] in old:
            row['title']=old[row['id']]['title'];row['acceptance']=old[row['id']]['acceptance'];row['status']=old[row['id']]['status'];row['evidence']=old[row['id']].get('evidence',[])
            if old[row['id']].get('status')=='completed' or 'approach' not in row:
                if 'approach' in old[row['id']]:row['approach']=old[row['id']]['approach']
                else:row.pop('approach',None)
            if 'verification' in old[row['id']]:row['verification']=old[row['id']]['verification']
            elif 'verification' in row:row.update(status='pending',evidence=[])
    # A compact user goal may omit the word "model"; the planner can infer the
    # deliverable from the materials. Once established, replanning cannot turn
    # model/tool development into an analysis-only completion.
    previous_deliverable=(previous or {}).get('deliverable')
    if model_goal(objective) or previous_deliverable=='model':value['deliverable']='model'
    elif previous_deliverable=='tool' and value.get('deliverable')!='model':value['deliverable']='tool'
    else:value.setdefault('deliverable','analysis')
    return value

def method_changes(previous, current, episode):
    old={row['id']:row for row in (previous or {}).get('steps',[])}
    return [{'episode':episode,'step_id':row['id'],'previous':old[row['id']].get('approach',''),
             'approach':row['approach'],'acceptance':row['acceptance']}
            for row in current.get('steps',[]) if row['id'] in old and row.get('approach')
            and old[row['id']].get('status')!='completed'
            and row['approach']!=old[row['id']].get('approach','')]


def create(root, profile, objective, limits):
    if profile.harness not in NATIVE_EXPERT_HARNESSES:raise ValueError('Select a native Scientist harness')
    store=GoalStore.create(root,GoalSpec(objective,CRITERIA,**limits),profile)
    write_json(store.root/'protocol.json',{'kind':DRIVER,'delivery_schema':DELIVERY_SCHEMA,'scientific_acceptance':sc_acceptance.POLICY})
    (store.root/'workspace').mkdir()
    return store

def verified_checkpoint(workspace, checkpoint, receipts, *, require_completed_steps=True):
    if not checkpoint or not receipts:return False
    if require_completed_steps and not checkpoint['completed_steps']:return False
    for row in checkpoint['artifacts']:
        if file_hash(safe_file(workspace,row['path']))!=row['sha256']:return False
    return bool(checkpoint['artifacts'])

class AutonomousResearch:
    def __init__(self, store, credential, runtimes, *, consult=None, executor=None, tools=None):
        self.store=store;self.credential=credential;self.runtimes=runtimes
        self.consult_override=consult;self.executor_override=executor;self.extra_tools=tools

    async def consult(self, role, context, profile, relay, folder):
        if self.consult_override:return await self.consult_override(role,context)
        from .scientist_role_output import specification, request_fields, output_content
        spec = specification(role, profile.model.protocol, model_acceptance=role=='SC' and bool(context.get('scientific_acceptance',{}).get('models')), study_acceptance=role=='SC' and context.get('scientific_acceptance',{}).get('study_evidence_required',False)) if hasattr(getattr(relay, 'policy', None), 'role_output') else None
        output_spec = spec
        role_prompt=sc_acceptance.prompt_for_review(context) if role=='SC' else ROLES[role]
        instruction=SCIENTIST_PURPOSE+role_prompt+' Respond in the language of the research objective.'
        if role=='SC' and context.get('scientific_acceptance',{}).get('models'):
            instruction += ' Exact response schema: '+json.dumps(sc_acceptance.model_schema(),ensure_ascii=False)
        if role=='CB':
            instruction += (' For each step, choose verification: artifact for documents; execution for computed analysis; '
                'data for checked data splits; model for fitted and measured models; revision for a measured descendant; '
                'review for independent approval; delivery for the final package. Host records must support completion. ')
            instruction += (' Include deliverable in the plan JSON: model, tool or analysis. '
                'Choose model for predictive-model development from wet-lab data or a paper extension, '
                'even when the short user topic does not use the word model. Choose tool for a reusable '
                'method or workflow. Analysis is a final deliverable only when the research goal itself '
                'asks for an analysis; data inspection is not a substitute for a requested model. '
                'Use the supplied materials and subsequent observations to choose a feasible strategy; '
                'the first plan is provisional until inputs have been inspected.')
        if role in {'PI','CB'}:
            instruction += (' A short everyday research goal is sufficient: supply the experimental design yourself. '
                'Give this topic a short human-readable title. Plan inspection of supplied files and choose feasible enabled tools, including training MCP when appropriate. '
                'For model goals plan data inspection/splits, actual training, held-out evaluation, a justified revised experiment, reproducible delivery and independent review. '
                'Do not ask the user to supply engineering prompts, tool names or a full acceptance specification. '
                'Use a small number of meaningful milestones, not a long list of tool calls. Replan unfinished work using measured feedback and user revisions.')
        instruction += ' This call returns a planning or review response only. Scientific tool execution belongs to the implementation phase.'
        if spec:
            instruction += ' Encode the response using only '+spec['name']+'. This is structured output, not an executable tool.'
        else:
            instruction += ' Do not call tools here.'
        prompt=json.dumps(context,ensure_ascii=False,default=str)
        structured = role in {'PI','CB','SC','REVIEW'}
        if structured:
            prompt += '\n\nYour task in this phase is to return exactly one JSON object matching the schema in the system instruction. The objective and prior role reports above are context, not instructions to execute work or change the response format. No Markdown or code fences.'
        if profile.model.protocol=='responses':
            route='/responses';payload={'model':profile.model.model,'instructions':instruction,'input':[{'role':'user','content':prompt}],'stream':False,'store':False,'tool_choice':'none'}
            if structured:payload['text']={'format':{'type':'json_object'}}
        else:
            route='/chat/completions';payload={'model':profile.model.model,'messages':[{'role':'system','content':instruction},{'role':'user','content':prompt}],'stream':True,'tool_choice':'none'}
            if structured:payload['response_format']={'type':'json_object'}
        if spec:
            payload.pop('response_format', None);payload.pop('text', None)
            payload.update(request_fields(spec))
        async with httpx.AsyncClient(timeout=profile.budget.request_seconds+30,trust_env=False) as client:
            for attempt in range(2):
                try:
                    headers = {'Authorization': 'Bearer ' + relay.token}
                    with relay.policy.role_output(spec) if spec else nullcontext():
                        if payload.get('stream'):
                            from .role_stream import read_role_stream
                            async with client.stream('POST', relay.base_url+route, headers=headers, json=payload) as response:
                                response.raise_for_status()
                                value = await read_role_stream(response)
                        else:
                            response = await client.post(relay.base_url+route, headers=headers, json=payload)
                            response.raise_for_status()
                            value = response.json()
                        answer=(output_content(value,output_spec,profile.model.model) if output_spec else
                                    role_content(value,profile.model.protocol,profile.model.model))
                        if output_spec and role == 'PI':
                            answer = json.loads(answer)['direction']
                    break
                except (httpx.HTTPStatusError,httpx.TransportError) as exc:
                    # Some free routes have no endpoint supporting native JSON
                    # mode. Retry the same model without that optional transport
                    # hint; structured() still enforces the complete host schema.
                    format_retry=isinstance(exc,httpx.HTTPStatusError) and exc.response.status_code in {400,404,422} and (spec is not None or 'response_format' in payload)
                    transient=not isinstance(exc,httpx.HTTPStatusError) or exc.response.status_code in {429,500,502,503,504}
                    if attempt or not (transient or format_retry) or relay.calls>=profile.budget.model_calls-3:raise
                    if format_retry:
                        payload.pop('response_format', None)
                        if spec:
                            instruction = instruction.replace(' Encode the response using only '+spec['name']+'. This is structured output, not an executable tool.', ' Return exactly one JSON object; do not call tools.')
                            if profile.model.protocol=='responses':payload['instructions']=instruction
                            else:payload['messages'][0]['content']=instruction
                            for key in ('tools','parallel_tool_calls'):payload.pop(key,None)
                            payload['tool_choice']='none'
                            spec=None
                    await asyncio.sleep(.5)
        return answer

    async def structured(self, role, context, profile, relay, folder, validator):
        """One bounded format repair within the existing model-call budget."""
        current=context
        for attempt in range(2):
            from .scientist_role_output import RoleOutputError
            problem=None
            try:raw=await self.consult(role,current,profile,relay,folder)
            except RoleOutputError as exc:raw=exc.raw;problem=exc
            self.activity(folder,role,raw)
            if problem is None:
                try:return validator(raw)
                except (ValueError,KeyError,TypeError) as exc:problem=exc
            if attempt or relay.calls>=profile.budget.model_calls-3:raise problem
            current={**context,'format_repair':{'previous_response':raw[:12000],
                'validation_error':str(problem)[:400],
                'instruction':'Reformat the same research content as one valid JSON object with the required fields. Do not invent results, change the objective or execute any tools.'}}
        raise AssertionError('Bounded structured request')

    def activity(self, folder, role, text):
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/'activity.json';rows=json.loads(path.read_text()) if path.exists() else []
        rows.append({'role':role,'content':text[:12000],'timestamp':time.time()});write_json(path,rows)

    def record_development_progress(self, event, development):
        """Expose host-measured generations without waiting for an episode to end."""
        if event.get('status')=='completed' and str(event.get('tool','')).rsplit('__',1)[-1] in {'evaluate_research_model','train_research_model'}:
            summary=development.summary()
            self.store.mutate(lambda state:state['progress'].update(development=summary))

    async def episode(self, index, profile, state, context_prompt):
        started=time.monotonic();folder=self.store.root/'episodes'/f'cycle_{index:03d}';folder.mkdir(parents=True,exist_ok=True)
        state=apply_revisions(self.store)
        protocol_path=self.store.root/'protocol.json'
        protocol=json.loads(protocol_path.read_text()) if protocol_path.exists() else {'kind':DRIVER,'delivery_schema':DELIVERY_SCHEMA}
        if protocol.get('scientific_acceptance')!=sc_acceptance.POLICY:
            write_json(protocol_path,{**protocol,'scientific_acceptance':sc_acceptance.POLICY})
        workspace=self.store.root/'workspace';progress=copy.deepcopy(state['progress']);events=[];relay=None;tools=[];service=None
        from .research_data_store import store_for_goal
        data_store=store_for_goal(self.store.root)
        if data_store is not None:
            data_store.protect_inputs(progress.get('inputs',[]),prior_agent_execution=bool(state.get('cycles')))
        context={'objective':state['definition']['goal']['objective'],'progress':progress,'previous_episodes':[{k:r.get(k) for k in ['summary','status']} for r in state['cycles'][-4:]],'workspace_files':[str(p.relative_to(workspace)) for p in workspace.rglob('*') if p.is_file()][:200]}
        from .research_input_index import input_descriptions
        context['supplied_materials'] = input_descriptions(workspace, progress.get('inputs', []),source_reader=data_store.input_bytes if data_store else None)
        context['research_topic']=topic_context()
        from mcp_tools.tools.runtime_tool_policy import filter_tools
        from .session_boundary import UNSAFE_TOOLS
        if self.extra_tools is None:
            from mcp_tools.tools.tools_agent_hub import get_tools
            available = get_tools()
        else: available = self.extra_tools
        scientific = [t for t in filter_tools(available)[0] if t.name not in UNSAFE_TOOLS]
        tool_index = ToolDisclosure({t.name: t for t in scientific}, schema=lambda t: {}, allowed=lambda t: True)
        preview = tool_index.metadata('search_tools', {'query': context['objective'][:200], 'limit': 12})
        if not preview.get('tools') and not preview.get('categories'):
            preview = tool_index.metadata('search_tools', {'limit': 12})
        context['available_tools'] = {'count': len(scientific), 'preview': preview.get('tools', []),
                                      'categories': preview.get('categories', []),
                                      'discovery': GUIDANCE}
        context['resources'] = {'online':os.getenv('VENUS_SCIENTIST_ONLINE')=='1', 'budget':state['definition']['goal'],
            'training':'Use enabled training MCP for platform PLMs; online only ESM2-8M. Isolated Python supports custom statistical and learning algorithms.'}
        development=Development(self.store.root/'development',data_store=data_store)
        from .research_attempt_resolution import training_progress, training_snapshot
        training_before = training_snapshot(development)
        context['model_development']={'prior_generations':development.summary(),
            'guidance_tool':'get_research_strategy_guide',
            'workflow':'Prepare checked data, fit a baseline, measure a justified revision, then submit independent Review.',
            'evidence_scope':'Host-owned datasets expose training rows only. The host computes validation feedback; final tests are reserved for independent Review. This does not establish prospective or out-of-domain validation.'}
        from .research_data_memory import prepared_dataset_context
        prepared = prepared_dataset_context(workspace, self.store.root/'episodes',data_store=data_store)
        if prepared is not None:
            context['model_development']['prepared_dataset'] = prepared
        if model_goal(context['objective']) or progress.get('plan',{}).get('deliverable')=='model':
            context['model_development']['next_action']=development_next_action(development.summary(),progress,prepared,materials=context['supplied_materials'])
        policy=ScientistGatewayPolicy(model=profile.model.model,mode='science_agent',scientific_tools=set(),max_calls=profile.budget.model_calls,temperature=profile.budget.temperature,max_tokens=profile.budget.output_tokens,thinking=profile.budget.thinking)
        gateway=EvidenceGateway
        if profile.model.protocol=='responses':
            from .responses_gateway import ResponsesGateway, ResponsesPolicy
            gateway=ResponsesGateway;policy=ScientistResponsesPolicy.from_profile(profile,[])
        elif profile.harness=='expert-codex':
            from .responses_bridge import ResponsesChatBridge
            gateway=ResponsesChatBridge
        elif profile.harness=='expert-claude-code':
            from .messages_bridge import MessagesBridge
            gateway=MessagesBridge
        class AwaitUser(Exception):
            pass
        result={'status':'failed' ,'summary':'Research episode did not complete','progress':progress}
        from .research_timeline import ResearchTimeline, ensure_project_readme
        timeline = ResearchTimeline(folder, index+1, workspace)
        def event_sink(event):
            events.append(event)
            if event.get('kind')=='research_tool_status':
                timeline.tool(event)
                self.record_development_progress(event, development)
        def phase(role):
            timeline.phase(role)
        try:
            async with gateway(policy=policy,upstream_url=profile.model.base_url,api_key=self.credential,output=folder/'model_requests.jsonl',timeout_s=profile.budget.request_seconds,stop_on_rate_limit=True) as relay:
                phase('PI')
                context['PI']=await self.structured('PI',context,profile,relay,folder,lambda value:value)
                phase('CB')
                plan=await self.structured('CB',context,profile,relay,folder,lambda raw:parse_plan(raw,progress.get('plan'),objective=context['objective']))
                changes = method_changes(progress.get('plan'), plan, index)
                context['plan']=plan;progress['plan']=plan
                write_json(folder/'plan.json',plan)
                if changes:
                    progress.setdefault('method_revisions', []).extend(changes)
                    context['progress']['method_revisions'] = progress['method_revisions']
                self.store.mutate(lambda s:s['progress'].update(plan=plan, method_revisions=progress.get('method_revisions', [])))
                ensure_project_readme(workspace, context['objective'], plan)
                context['planning_questions']=plan['needs_input']
                if plan.get('deliverable')=='model' and data_store is None:
                    data_store=store_for_goal(self.store.root,force=True)
                    data_store.protect_inputs(progress.get('inputs',[]),prior_agent_execution=bool(state.get('cycles')))
                    development.data_store=data_store
                service=ResearchWorkspace(workspace,folder/'tools',plan,development_root=self.store.root/'development',data_store=data_store)
                from .interactions import INTERACTION_NAMES
                from mcp_tools.tools.tool_surface import WORKSPACE_PRIMITIVES
                tools=scientific+service.tools()
                if plan.get('deliverable')=='model':
                    context['model_development']['next_action']=development_next_action(development.summary(),progress,prepared,materials=context['supplied_materials'])
                    repair = repair_source_context(workspace, progress.get('execution_feedback'),
                        input_fields=(prepared or {}).get('model_interface', {}).get('input_fields', {}), task=(prepared or {}).get('task', ''))
                    if repair is not None:context['model_development']['repair_source'] = repair
                focus=selected_tools(context,tools)
                context['model_development']['selected_tools']=focus
                names=CONTROLS | WORKSPACE_PRIMITIVES | INTERACTION_NAMES | {'submit_research_checkpoint'} | set(focus)
                context['tools'] = sorted(names)
                if profile.model.protocol=='responses':policy.allowed={'mcp__venusfactory__'+n for n in names}
                else:policy.scientific_tools=frozenset(names)
                remaining=profile.budget.model_calls-relay.calls
                if remaining<4:raise ValueError('No budget left for execution and review')
                native=replace(profile,mode='science_agent',harness=NATIVE_EXPERT_HARNESSES[profile.harness],budget=replace(profile.budget,model_calls=remaining-3,wall_seconds=max(1,profile.budget.wall_seconds-(time.monotonic()-started))))
                instructions=execution_instructions(context,native.budget.tool_calls)
                request={'session_id':self.store.root.name,'profile':native.public(),'runtimes':self.runtimes,'research_focus':focus,'workspace':str(workspace),'control':str(folder),'messages':[],'text':json.dumps(context,ensure_ascii=False),'instructions':instructions,'base_url':relay.base_url,'token':relay.token,'budget':native.public()['budget'],'lang':'zh' if any('\u4e00'<=c<='\u9fff' for c in context['objective']) else 'en'}
                phase('MLS')
                if self.executor_override:await self.executor_override(request,service,events)
                else:
                    from .web_native_worker import run
                    request['audit_required']=True
                    from .research_handoff import native_phase
                    with native_phase(policy) as handoff:
                        await run(request,tools=tools,event_sink=event_sink,checkpoint_handoff=handoff)
                from .execution_followup import execution_followup, execution_usage
                followup = execution_followup(
                    request, native, events, checkpoint=service.checkpoint,
                    remaining_calls=profile.budget.model_calls-relay.calls-3,
                    remaining_seconds=max(0,profile.budget.wall_seconds-(time.monotonic()-started)))
                if followup is not None:
                    retry_folder=folder/'execution_retry';retry_folder.mkdir()
                    followup['control']=str(retry_folder)
                    continuation = json.loads(followup['text'])
                    current_development = development.summary()
                    continuation['model_development']['prior_generations'] = current_development
                    prepared = prepared_dataset_context(workspace, self.store.root/'episodes',data_store=data_store)
                    continuation['model_development']['next_action'] = development_next_action(current_development, progress, prepared, materials=continuation['supplied_materials'])
                    if prepared is not None:
                        continuation['model_development']['prepared_dataset'] = prepared
                    else:
                        continuation['model_development'].pop('prepared_dataset', None)
                    continuation['execution_receipts'] = service.receipts[-6:]
                    repair = repair_source_context(workspace, feedback_state(events, workspace, progress.get('execution_feedback'), development=development),
                        input_fields=(prepared or {}).get('model_interface', {}).get('input_fields', {}), task=(prepared or {}).get('task', ''))
                    if repair is not None:continuation['model_development']['repair_source'] = repair
                    else:continuation['model_development'].pop('repair_source', None)
                    focus = selected_tools(continuation, tools)
                    continuation['model_development']['selected_tools'] = focus
                    names = CONTROLS | WORKSPACE_PRIMITIVES | INTERACTION_NAMES | {'submit_research_checkpoint'} | set(focus)
                    continuation['tools'] = sorted(names)
                    if profile.model.protocol == 'responses':
                        policy.allowed = {'mcp__venusfactory__'+name for name in names}
                    else:
                        policy.scientific_tools = frozenset(names)
                    followup.update(text=json.dumps(continuation,ensure_ascii=False), research_focus=focus,
                                    instructions=execution_instructions(continuation,followup['budget']['tool_calls']))
                    write_json(retry_folder/'reason.json', {'reason':'native_return_without_checkpoint',
                        'maximum_retries':1, 'used':execution_usage(events),
                        'remaining_execution_calls':followup['budget']['tool_calls']})
                    if self.executor_override:
                        await self.executor_override(followup,service,events)
                    else:
                        from .research_handoff import native_phase
                        with native_phase(policy) as handoff:
                            await run(followup,tools=tools,event_sink=event_sink,checkpoint_handoff=handoff)
                awaiting=next((e for e in events if e.get('kind')=='interaction'),None)
                if awaiting:
                    from .interactions import Interaction
                    gate=Interaction.model_validate(awaiting['interaction']).model_dump()
                    progress.update(pending_interaction=gate, questions=[q['question'] for q in gate['questions']] or [gate['message']], research_completed=False)
                    progress.pop('reviewed_episode',None)
                    self.activity(folder,'MLS',json.dumps(gate,ensure_ascii=False))
                    raise AwaitUser()
                execution=next((e for e in reversed(events) if e.get('kind')=='completed'),None)
                if not execution:raise ValueError('Native executor did not complete')
                if not (execution.get('event_audit') or {}).get('passed'):raise ValueError('Scientific execution evidence was not validated')
                self.activity(folder,'MLS',execution['text'])
                checkpoint=service.checkpoint
                context['execution']={'report':execution['text'],'checkpoint':checkpoint,'receipts':service.receipts,'tool_results':[e for e in events if e.get('kind')=='tool_result'][-20:]}
                previews=[]
                for row in (checkpoint or {}).get('artifacts',[]):
                    p=safe_file(workspace,row['path']);preview={'path':row['path'],'sha256':row['sha256']}
                    if p.suffix in {'.json','.txt','.md','.csv','.py'} and p.stat().st_size<=40000:preview['content']=p.read_text(errors='replace')
                    previews.append(preview)
                context['artifact_previews']=previews
                current_feedback=feedback_state(events,workspace,progress.get('execution_feedback'),development=development)
                context['execution']['failures']=current_feedback['errors']
                context['execution']['superseded_attempts']=current_feedback.get('superseded', [])
                step_checks=checkpoint_steps(plan,checkpoint,workspace=workspace,receipts=service.receipts,
                    experiments=development.records(),progress=progress)
                context['execution']['step_checks']=step_checks
                progress['step_checks']=step_checks
                phase('SC')
                progress.pop('reviewed_episode',None)
                progress['research_completed']=False
                sc_context=sc_acceptance.evidence_context(context,service.proposals,development)
                validator = critic_decision
                if sc_context['scientific_acceptance']['models']:
                    validator = sc_acceptance.parse_model_review
                elif sc_context['scientific_acceptance'].get('study_evidence_required'):
                    def validator(raw):
                        return sc_acceptance.parse_study_review(raw, sc_context)
                review=await self.structured('SC',sc_context,profile,relay,folder,validator)
                write_json(folder/'review.json',review)
                acceptance=sc_acceptance.record(folder,review,sc_context,goal_root=self.store.root)
                progress['scientific_acceptance']=acceptance
                if not acceptance['passed']:
                    review={**review,'decision':'needs_revision','issues':acceptance['issues']}
                    progress['sc_revision']=sc_acceptance.revision_feedback(acceptance,sc_context,review,index)
                    progress.pop('reviewed_episode',None)
                    progress['research_completed']=False
                elif acceptance['approved_models'] or not progress.get('sc_revision',{}).get('requests'):
                    progress.pop('sc_revision',None)
                checkpoint_valid=verified_checkpoint(workspace,checkpoint,service.receipts,require_completed_steps=False)
                request_valid=audit_requests(folder/'model_requests.jsonl')['passed']
                host_issues=[row['error'] for row in context['execution']['failures']]
                if not checkpoint_valid:host_issues.append('Submit a checkpoint with unchanged artifacts and execution receipts.')
                if not request_valid:host_issues.append('Model request evidence did not pass verification.')
                reviewable=not host_issues
                critic_ready=review['decision']=='ready_for_host_validation'
                step_issues=[f"Step {row['id']}: {issue}" for row in step_checks for issue in row['issues']]
                host_issues.extend(step_issues)
                host_review={'passed':not host_issues,'execution_passed':reviewable,'issues':host_issues,
                             'steps':step_checks,'model_decision':review['decision'],'scientific_acceptance_passed':acceptance['passed']}
                write_json(folder/'host_review.json',host_review)
                progress['host_review']=host_review
                if host_issues:
                    review={'decision':'needs_revision','summary':'Execution needs revision: '+failure_summary(host_issues[0]),
                            'issues':host_issues}
                    self.activity(folder,'SC',json.dumps({'source':'automatic_checks',**review},ensure_ascii=False))
                accepted_steps={row['id'] for row in step_checks if row['passed']}
                ready=critic_ready and reviewable and bool(accepted_steps)
                if ready:
                    accepted_plan=copy.deepcopy(plan)
                    for step in accepted_plan['steps']:
                        if step['id'] in accepted_steps:
                            step.update(status='completed',evidence=checkpoint['artifacts'])
                    plan=accepted_plan
                    progress['artifacts']=checkpoint['artifacts']
                    progress['reviewed_episode']=index
                # SC must accept usefulness before technical Review. Pending
                # technical-review plan steps themselves are not a circular gate.
                if reviewable and acceptance['passed']:
                    reviews={r['name']:r for r in progress.get('publication_reviews',[])}
                    for bundle in service.proposals:
                        phase('REVIEW')
                        async def independent(candidate):
                            # A fresh tool-free request: no PI/SC messages or their approval.
                            return await self.consult('REVIEW',candidate,profile,relay,folder)
                        receipt=await review_proposal(bundle,reviewer=independent,control=folder/'admission',runner=service.runner)
                        candidate=json.loads((bundle/'manifest.json').read_text())
                        row={'id':candidate['id'],'name':candidate['name'],'status':receipt['status'],'issues':receipt['issues'],
                             'summary':(receipt.get('independent') or {}).get('summary',''),
                             'receipt':str((bundle/'admission.json').relative_to(self.store.root)),'sha256':file_hash(bundle/'admission.json')}
                        if (bundle/'submission.json').is_file():
                            row['proposal_file']=json.loads((bundle/'submission.json').read_text()).get('proposal_file')
                        row['review_stage']='independent' if receipt.get('independent') is not None else 'automatic'
                        if receipt['status']=='approved' and audit_requests(folder/'model_requests.jsonl')['passed']:
                            try:
                                delivery=None
                                scientific_binding=sc_acceptance.bind_delivery(bundle,acceptance)
                                if candidate.get('development') or candidate['review_spec']['kind']=='tool':
                                    delivery=build_delivery(workspace,bundle,receipt,goal_id=self.store.root.name)
                                    if scientific_binding:delivery['scientific_acceptance']=scientific_binding
                                kwargs={'library':self.store.root/'private_capabilities'}
                                published=publish_proposals([bundle],self.store.root.name,**kwargs)
                                progress['capabilities']=list({r['id']:r for r in [*progress.get('capabilities',[]),*published]}.values())
                                if delivery:
                                    progress['deliveries']=[r for r in progress.get('deliveries',[]) if r['candidate_id']!=delivery['candidate_id']]+[delivery]
                                    progress['selected_deliveries']=selected_deliveries(progress['deliveries'])
                            except Exception as exc:
                                row.update(status='needs_revision',issues=['Registration: '+str(exc)[:500]])
                        elif receipt['status']=='approved': row.update(status='needs_revision',issues=['Model response audit did not pass'])
                        reviews[row['name']]=row
                        progress['publication_reviews']=list(reviews.values())
                        self.activity(folder,'REVIEW',json.dumps(row,ensure_ascii=False))
                        self.store.mutate(lambda s:s['progress'].update(publication_reviews=progress['publication_reviews']))
                questions=(checkpoint or {}).get('needs_input',[])
                progress.update(plan=plan,questions=questions,last_review=review,research_completed=ready and all(s['status']=='completed' for s in plan['steps']))
                progress['development']=development.summary()
                if plan.get('deliverable')=='model' and not has_delivery(progress.get('deliveries',[]),'model'):
                    ready=False;progress['research_completed']=False
                    progress['delivery_pending']='A measured baseline, revised model, independent admission and model delivery are required'
                elif plan.get('deliverable')=='tool' and (not progress.get('capabilities') or not has_delivery(progress.get('deliveries',[]),'method')):
                    ready=False;progress['research_completed']=False
                    progress['delivery_pending']='The reusable tool needs independent review and a registered delivery'
                else:progress.pop('delivery_pending',None)
                if any(r['status']!='approved' for r in progress.get('publication_reviews',[])): ready=False
                result.update(status='completed' if ready else 'needs_revision',summary=review['summary'],needs_input=bool(questions))
        except AwaitUser:
            result.update(status='needs_revision',summary='Waiting for your response',needs_input=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result.update(summary='Research episode needs revision: '+type(exc).__name__+': '+str(exc)[:400],error_type=type(exc).__name__)
            write_json(folder/'failure.json',{'error_type':type(exc).__name__,'message':str(exc)[:400]})
        finally:
            timeline.finish(result['status'])
            progress['development']=development.summary()
            progress['training_progress'] = training_progress(development, training_before, progress.get('training_progress'))
            if service is not None and service.latest_submission:
                progress['review_submission']=service.latest_submission
            progress['execution_feedback']={'iteration':index+1, **feedback_state(events,workspace,progress.get('execution_feedback'),development=development)}
            if model_goal(context['objective']) and not has_delivery(progress.get('deliveries',[]),'model'):
                progress['delivery_pending']='Continue from measured generations; model review and delivery are not yet complete'
            write_json(folder/'events.json',events)
        calls=audit_requests(folder/'model_requests.jsonl') if (folder/'model_requests.jsonl').exists() else {'model_calls':0,'passed':False}
        if result['status']=='completed' and not calls['passed']:
            result.update(status='failed',summary='Model response audit did not complete')
            progress.pop('reviewed_episode',None)
        from .model_availability import model_service_blocker
        blocker=model_service_blocker(folder/'model_requests.jsonl')
        if not blocker and progress.get('training_progress', {}).get('intervention_required'):
            blocker = {'source': 'research_progress', 'code': 'unchanged_training_failure',
                'message': 'Training repeated the same failing implementation for three research cycles without a new measured model. '
                    'The project and remaining budget are saved. Revise the implementation or model configuration before resuming.'}
        if blocker and result['status']!='completed':
            progress['execution_blocker']=blocker
            result.update(execution_blocker=blocker,summary=blocker['message'])
            write_json(folder/('model_service.json' if blocker['source']=='model_gateway' else 'training_stagnation.json'),blocker)
        elif not blocker:
            progress.pop('execution_blocker',None)
        from .execution_followup import execution_usage
        result['usage']={'model_calls':calls['model_calls'],
            'tool_calls':execution_usage(events)['tool_calls'],
            'wall_seconds':time.monotonic()-started}
        return result

    def evaluate(self,state):
        progress=state['progress'];plan=progress.get('plan',{});checks={k:{'passed':False} for k in state['definition']['goal']['criteria']}
        if state.get('pending_research_updates'):return {'criteria':checks}
        if not plan.get('steps') or not all(s.get('status')=='completed' for s in plan['steps']) or 'reviewed_episode' not in progress:return {'criteria':checks}
        if plan.get('deliverable')=='tool' and not progress.get('capabilities'):
            return {'criteria':checks,'delivery_pending':True}
        protocol=self.store.root/'protocol.json'
        protocol_data=json.loads(protocol.read_text()) if protocol.exists() else {}
        modern_delivery=protocol_data.get('delivery_schema')==DELIVERY_SCHEMA
        strict_sc=protocol_data.get('scientific_acceptance')==sc_acceptance.POLICY
        if strict_sc and (progress.get('sc_revision') or progress.get('last_review',{}).get('decision','ready_for_host_validation')!='ready_for_host_validation'):
            return {'criteria':checks,'scientific_acceptance_pending':True}
        if plan.get('deliverable')=='model' or model_goal(state['definition']['goal']['objective']) or (plan.get('deliverable')=='tool' and modern_delivery):
            kind='method' if plan.get('deliverable')=='tool' and not model_goal(state['definition']['goal']['objective']) else 'model'
            if not has_delivery(progress.get('deliveries',[]),kind):return {'criteria':checks,'delivery_pending':True}
            for delivery in selected_deliveries(progress['deliveries']):
                if delivery['candidate_id'] not in {c['id'] for c in progress.get('capabilities',[])}:return {'criteria':checks,'delivery_pending':True}
                if strict_sc and delivery.get('kind')=='model' and not sc_acceptance.verify_delivery(self.store.root,delivery):
                    return {'criteria':checks,'scientific_acceptance_pending':True}
                for artifact in delivery['artifacts']:
                    try:
                        if file_hash(safe_file(self.store.root/'workspace',artifact['path']))!=artifact['sha256']:return {'criteria':checks,'delivery_pending':True}
                    except (OSError,ValueError):return {'criteria':checks,'delivery_pending':True}
        for step in plan['steps']:
            for row in step.get('evidence',[]):
                try:
                    if file_hash(safe_file(self.store.root/'workspace',row['path']))!=row['sha256']:return {'criteria':checks}
                except (OSError,ValueError):return {'criteria':checks}
        index=progress['reviewed_episode'];review=self.store.root/'episodes'/f'cycle_{index:03d}'/'review.json'
        if not review.exists() or json.loads(review.read_text())['decision']!='ready_for_host_validation':return {'criteria':checks}
        admission=progress.get('publication_reviews',[])
        admission_pending=any(r['status']!='approved' for r in admission)
        for row in admission:
            try:
                if file_hash(safe_file(self.store.root,row['receipt']))!=row['sha256']:return {'criteria':checks}
            except (OSError,ValueError):return {'criteria':checks}
        evidence=[{'path':r['receipt'],'sha256':r['sha256']} for r in admission]+[{'path':str(review.relative_to(self.store.root)),'sha256':file_hash(review)}]
        for step in plan['steps']:
            evidence += [{'path':'workspace/'+r['path'],'sha256':r['sha256']} for r in step['evidence']]
        checks={k:{'passed':True,'evidence':evidence} for k in checks}
        if admission_pending:
            # Preserve old immutable goal definitions; new goals carry a distinct
            # admission criterion so completed research remains visibly complete.
            key='capability_admission_resolved' if 'capability_admission_resolved' in checks else 'research_outputs_reviewed'
            checks[key]={'passed':False,'reason':'Reusable candidates need admission revision'}
        return {'criteria':checks,'research_completed':True,'admission_pending':admission_pending}

    async def run(self):
        return await run_goal(self.store,self.episode,self.evaluate,stop_on_uncheckpointed_failure=True)
