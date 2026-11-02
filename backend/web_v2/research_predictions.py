"""Sequence-compatible, checkpoint-backed report predictions through the MCP queue."""
import asyncio
import csv
import hashlib
import json
import os
from pathlib import Path
import time

from mcp_tools.tools.runtime_tool_policy import is_online_mode

ROOT = Path(__file__).resolve().parents[2]
GENERAL = {'Solubility', 'Subcellular Localization', 'Membrane Protein', 'Metal Ion Binding', 'Stability', 'Sortingsignal'}
ENZYME = {'Optimal Temperature', 'Optimal PH'}
LABELS = {'Solubility': '溶解度', 'Subcellular Localization': '亚细胞定位', 'Membrane Protein': '膜蛋白倾向',
          'Metal Ion Binding': '金属离子结合', 'Stability': '稳定性', 'Sortingsignal': '分选信号',
          'Optimal Temperature': '最适温度', 'Optimal PH': '最适 pH', 'Activity Site': '活性位点',
          'Binding Site': '结合位点', 'Conserved Site': '保守位点', 'Motif': '序列基序'}


def bounded_setting(name, default, maximum):
    try: return min(maximum, max(1, int(os.getenv(name, str(default)))))
    except ValueError: return default


def candidates(context, *, root=ROOT):
    """No immunogenicity or substrate-dependent kinetics inferred from sequence alone."""
    cfg = json.loads((root / 'src/constant.json').read_text())['web_ui']
    entry = context.get('entry') or {}
    enzyme = any(c.get('commentType') == 'CATALYTIC ACTIVITY' for c in entry.get('comments', []))
    tasks = GENERAL | (ENZYME if enzyme else set())
    groups = []
    for mapping, models, tool in [
        ('dataset_mapping_function', 'model_mapping_function', 'predict_protein_function'),
        ('residue_mapping_function', 'model_residue_mapping_function', 'predict_residue_function'),
    ]:
        for task, datasets in cfg[mapping].items():
            if tool == 'predict_protein_function' and task not in tasks: continue
            group = []
            for dataset in datasets:
                for model, key in cfg[models].items():
                    if tool == 'predict_residue_function' and key != 'esm2': continue
                    adapter = root / 'ckpt' / dataset / cfg['model_adapter_mapping_function'][key]
                    # Missing weights do not trigger large automatic downloads from a report.
                    if not (adapter / 'lr5e-4_bt12k_ga8.pt').is_file() or not (adapter / 'lr5e-4_bt12k_ga8.json').is_file(): continue
                    identity = hashlib.sha256(f'{tool}:{task}:{dataset}:{model}'.encode()).hexdigest()[:16]
                    from mcp_tools.tools.runtime_tool_policy import assert_tool_call_allowed
                    try: assert_tool_call_allowed(tool, {'task':task,'model_name':model,'adapter_path':str(adapter)})
                    except (ValueError,RuntimeError): continue
                    group.append({'id': identity, 'task': task, 'label_zh': LABELS.get(task, task), 'model': model,
                                  'dataset': dataset, 'tool': tool, 'adapter_path': str(adapter)})
            if group: groups.append(group)
    # Cover every property first, then add independent backbones/datasets for comparison.
    return [g[i] for i in range(max(map(len, groups), default=0)) for g in groups if i < len(g)]


def options():
    rows = candidates({})
    return {'available': not is_online_mode(), 'candidate_count': len(rows),
            'limit': bounded_setting('VENUS_REPORT_PREDICTION_LIMIT', 16, 32),
            'properties': list(dict.fromkeys(r['task'] for r in rows)),
            'backbones': list(dict.fromkeys(r['model'] for r in rows))}


def read_result(path):
    if not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError('Prediction output is unavailable')
    with path.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    if not rows: raise ValueError('Prediction output is empty')
    # Preserve model scores and their names; do not invent calibrated probabilities or units.
    return [{k: v for k, v in row.items() if k not in {'sequence', 'header'}} for row in rows[:1]]


async def queued_prediction(spec, sequence_file, output, on_status):
    from mcp_tools.scheduler.client import invoke
    if is_online_mode(): raise ValueError('Predictions are available in local mode')
    owner = os.environ.get('VENUS_QUEUE_OWNER')
    if not owner: raise ValueError('Predictions require an owned research job')
    return await invoke(spec['tool'], {'fasta_file': str(sequence_file), 'task': spec['task'],
        'model_name': spec['model'], 'adapter_path': spec['adapter_path'], 'output_file': str(output)},
        owner=owner, parent_job=os.environ.get('VENUS_QUEUE_JOB_ID'), on_status=on_status)


async def run_predictions(context, update, folder, *, execute=None):
    if is_online_mode(): raise ValueError('Predictions are available in local mode')
    execute = execute or queued_prediction
    sequence = context.get('sequence', '')
    if not sequence or not set(sequence) <= set('ACDEFGHIKLMNPQRSTVWY'):
        update([]); return []
    specs = candidates(context)[:bounded_setting('VENUS_REPORT_PREDICTION_LIMIT', 16, 32)]
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    fasta = folder / 'query.fasta'; fasta.write_text('>query\n' + sequence + '\n'); fasta.chmod(0o600)
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    rows = [{k: v for k, v in spec.items() if k != 'adapter_path'} | {'status': 'queued', 'sequence_sha256': digest} for spec in specs]
    update(rows)
    deadline = time.monotonic() + bounded_setting('VENUS_REPORT_PREDICTION_SECONDS', 900, 1800)
    for spec, row in zip(specs, rows):
        if time.monotonic() >= deadline:
            row.update(status='deferred', message='Report prediction time budget reached'); update(rows); continue
        if is_online_mode():
            row.update(status='deferred', message='Available in local mode'); update(rows); continue
        output = folder / (spec['id'] + '.csv')
        row.update(started=time.time())
        def progress(job):
            # A worker exiting is not proof of a usable prediction; validate its output below.
            row.update(status='running' if job['status'] == 'completed' else job['status'],
                       job_number=job.get('job_number'), position=job.get('position'))
            update(rows)
        try:
            value = await asyncio.wait_for(execute(spec, fasta, output, progress), timeout=min(240, max(1, deadline-time.monotonic())))
            if isinstance(value, str): value = json.loads(value)
            if not isinstance(value, dict) or value.get('status') != 'success': raise ValueError('No prediction result returned')
            row.update(status='completed', values=read_result(output), evidence_level='model_prediction', output_file=output.name)
        except asyncio.TimeoutError:
            row.update(status='deferred', message='Prediction time limit reached')
        except Exception:
            row.update(status='unavailable', message='Model output unavailable')
        row['finished'] = time.time(); update(rows)
    return rows


def report_appendix(rows, lang='en'):
    if not rows: return ''
    zh = lang == 'zh'
    lines = ['## ' + ('AI 模型预测' if zh else 'AI model predictions'),
             '模型推断，非实验测量；不同任务的原始分数不能直接比较。' if zh else 'Model predictions, not experimental measurements. Raw scores from different tasks are not directly comparable.',
             '| ' + ('性质 | 模型 / 数据集 | 输出 |' if zh else 'Property | Model / dataset | Output |'), '|---|---|---|']
    def cell(x): return str(x).replace('|', '\\|').replace('\n', ' ')[:1200]
    for row in rows:
        result = json.dumps(row.get('values'), ensure_ascii=False) if row['status'] == 'completed' else ('暂无结果' if zh else 'No result')
        lines.append(f"| {cell(row.get('label_zh') if zh else row['task'])} | {cell(row['model'])} / {cell(row['dataset'])} | {cell(result)} |")
    return '\n\n'.join(lines[:2]) + '\n\n' + '\n'.join(lines[2:])
