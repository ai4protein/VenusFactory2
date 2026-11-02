"""Small, durable dataflow observations; never include storage or credential metadata."""
import math
import re


def summarize(value):
    kind = value.get('kind', 'table')
    result = {'kind': kind}
    if kind == 'model':
        model = value.get('model') or {}
        result['model'] = {key: model[key] for key in ('id', 'name', 'metrics') if key in model}
    elif kind == 'table':
        rows = value.get('rows', [])
        columns = list(dict.fromkeys(key for row in rows for key in row))
        visible = [key for key in columns if not re.search(r'password|secret|token|api_key|authorization|file_path|out_dir', key, re.I)][:12]
        def cell(v):
            if v is None or isinstance(v, (bool, int)): return v
            if isinstance(v, float): return v if math.isfinite(v) else None
            if not isinstance(v, str): return '[structured value]'
            if re.match(r'^(?:/|[a-zA-Z]:[\\/])', v): return '[file]'
            return v[:180] + ('…' if len(v) > 180 else '')
        result.update(rows=len(rows), columns=visible, column_count=len(columns),
                      preview=[{k: cell(row.get(k)) for k in visible} for row in rows[:3]])
    else:
        result['files'] = 1
    return result


def validate_columns(node, data, model=None):
    """Check actual upstream columns before invoking a tool or loading weights."""
    if not data or data.get('kind') != 'table': return
    rows = data.get('rows', [])
    columns = set.intersection(*(set(row) for row in rows)) if rows else set()
    cfg = node.get('config') or {}
    mapping = cfg.get('columns') or {}
    if not isinstance(mapping, dict) or any(not isinstance(v, str) for v in mapping.values()):
        raise ValueError('Column mapping must map parameter names to column names')
    needed = set(mapping.values())
    adapter = (model or {}).get('adapter')
    if node['kind'] == 'train':
        needed.update(cfg.get('features') or [])
        needed.update(cfg.get(key) for key in ('target', 'sequence_column', 'group_column') if cfg.get(key))
    elif adapter == 'personal':
        needed.update(model.get('features') or [])
        if model.get('sequence_column'): needed.add(model['sequence_column'])
    elif adapter in {'descriptors', 'checkpoint', 'function', 'residue', 'evolution'}:
        needed.add(cfg.get('sequence_column') or 'sequence')
    elif adapter == 'registered-tool' and not model.get('standalone'):
        needed.add(cfg.get('sequence_column') or 'sequence')
    missing = sorted(needed - columns)
    if missing:
        raise ValueError((node.get('label') or node['id']) + ': Missing input columns: ' + ', '.join(missing))
