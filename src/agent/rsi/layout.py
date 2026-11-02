"""Resolve current source files without rewriting recorded historical paths."""
from pathlib import PurePosixPath
from ._compat import MODULES


def source_relative(filename):
    name = PurePosixPath(filename)
    if name.name != filename or name.suffix != '.py':
        raise ValueError('Expected one RSI Python source filename')
    module = MODULES.get(name.stem)
    return 'src/agent/rsi/' + (module.replace('.', '/') + '.py' if module else filename)


def recorded_relative(implementation, filename):
    """An audit follows the path bound by its protocol, including pre-layout runs."""
    current, historical = source_relative(filename), 'src/agent/rsi/' + filename
    matches = [p for p in dict.fromkeys((current, historical)) if p in implementation]
    if len(matches) > 1:
        raise ValueError('Ambiguous recorded RSI source')
    # Missing paths stay missing; the caller retains its original error contract.
    return matches[0] if matches else historical


def recorded_path(implementation, relative):
    prefix = 'src/agent/rsi/'
    if relative.startswith(prefix) and '/' not in relative[len(prefix):]:
        return recorded_relative(implementation, relative[len(prefix):])
    return relative
