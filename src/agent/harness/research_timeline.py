"""Small, durable public progress records; never model reasoning or request bodies."""
import json
import time
from .host import write_json
from .session_boundary import public_value


def ensure_project_readme(workspace, objective, plan):
    """Start a truthful project landing page; never replace researcher-authored text."""
    from .research_workspace import safe_file
    from pathlib import Path
    workspace = Path(workspace)
    existing = [p for p in workspace.iterdir() if p.name.lower() == 'readme.md']
    if existing or not any(p.is_file() for p in workspace.rglob('*')): return
    path = safe_file(workspace, 'README.md', exists=False)
    zh = any('\u4e00' <= c <= '\u9fff' for c in objective)
    title = '研究项目' if zh else 'Research project'
    headings = ('研究目标','研究计划','输入材料') if zh else ('Objective','Research plan','Inputs')
    lines = ['# '+title, '', '## '+headings[0], '', objective, '', '## '+headings[1], '']
    lines += ['- '+row['title'] for row in plan.get('steps',[])]
    inputs = [p.relative_to(workspace).as_posix() for p in sorted((workspace/'inputs').rglob('*')) if p.is_file() and not p.is_symlink()]
    if inputs: lines += ['', '## '+headings[2], '', *['- `'+p+'`' for p in inputs[:100]]]
    # Exclusive creation preserves a concurrent authored README.
    try:
        with path.open('x',encoding='utf-8') as handle: handle.write('\n'.join(lines)+'\n')
    except FileExistsError: pass


class ResearchTimeline:
    def __init__(self, folder, episode, workspace):
        self.folder, self.episode, self.workspace = folder, episode, workspace
        self.stages, self.events = [], []

    def phase(self, role):
        now = time.time()
        if self.stages and self.stages[-1]['status'] == 'running':
            self.stages[-1].update(status='completed', ended_at=now)
        self.stages.append(dict(role=role, episode=self.episode, status='running', started_at=now))
        write_json(self.folder/'phase.json', dict(role=role, episode=self.episode, updated_at=now))
        write_json(self.folder/'stages.json', self.stages)

    def tool(self, event):
        row = public_value({k:event[k] for k in ('tool','status','call_id','job_id','job_number','position','progress','error_type') if k in event}, self.workspace)
        row.update(timestamp=time.time(), role=self.stages[-1]['role'] if self.stages else 'MLS', episode=self.episode)
        terminal = {'completed', 'failed', 'cancelled', 'interrupted'}
        match = None
        for index in range(len(self.events) - 1, -1, -1):
            previous = self.events[index]
            if (previous.get('tool'), previous.get('role')) != (row.get('tool'), row['role']):
                continue
            if row.get('call_id'):
                if previous.get('call_id') != row['call_id']:
                    continue
                # Legacy providers may restart their counter on a retry.
                if previous.get('status') in terminal and row.get('status') not in terminal:
                    break
            else:
                # Without an ID only the most recent unfinished operation can
                # receive progress. Finished calls must keep their own history.
                if index != len(self.events) - 1 or previous.get('call_id') or previous.get('status') in terminal:
                    break
                if row.get('job_id') and previous.get('job_id') not in (None, row['job_id']):
                    break
            match = index
            break
        if match is not None:
            previous = self.events[match]
            row = {**previous, **row, 'timestamp':previous['timestamp']}
            self.events[match] = row
        else:
            self.events.append(row)
        write_json(self.folder/'current_tool.json', row)
        write_json(self.folder/'timeline.json', self.events[-100:])

    def finish(self, status):
        if self.stages:
            self.stages[-1].update(status='completed' if status in {'completed','needs_revision'} else 'interrupted', ended_at=time.time())
            write_json(self.folder/'stages.json', self.stages)
