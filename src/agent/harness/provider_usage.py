"""Read only provider-reported token counts, including interrupted responses."""
import json


def reported_usage(raw: bytes, streaming: bool, *, responses=False):
    """Return known counters without interpreting text or inventing missing usage.

    SSE usage fields are snapshots, not increments. Later fields replace earlier
    ones; cache/reasoning breakdowns are already included in provider totals.
    """
    counters = {}
    def objects():
        if streaming:
            for line in raw.splitlines():
                if not line.startswith(b'data:'):
                    continue
                try:
                    yield json.loads(line[5:].strip())
                except (ValueError, UnicodeError):
                    continue
        else:
            try:
                yield json.loads(raw)
            except (ValueError, UnicodeError):
                return

    for value in objects():
        if not isinstance(value, dict):
            continue
        parent = value.get('response') if responses else value
        usage = parent.get('usage') if isinstance(parent, dict) else None
        if not isinstance(usage, dict):
            continue
        snapshot = {}
        for key, aliases in {
            'input_tokens': ('input_tokens', 'prompt_tokens'),
            'output_tokens': ('output_tokens', 'completion_tokens'),
            'total_tokens': ('total_tokens',),
        }.items():
            for alias in aliases:
                number = usage.get(alias)
                if type(number) is int and 0 <= number <= 2**63-1:
                    snapshot[key] = number
                    break
        if 'total_tokens' not in snapshot and any(key in snapshot for key in ('input_tokens','output_tokens')):
            counters.pop('total_tokens', None)
        counters.update(snapshot)
    return counters or None
