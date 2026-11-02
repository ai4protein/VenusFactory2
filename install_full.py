#!/usr/bin/env python3
"""One-command complete installation; --components selects optional families."""
import sys
from scripts.bootstrap.cli import main


def arguments(argv):
    args = list(argv)
    if not args or args[0].startswith('-'):
        args.insert(0, 'install')
    if any(a == '--profile' or a.startswith('--profile=') for a in args):
        raise ValueError('install_full.py uses profile full; use initialize.py for another profile')
    return [args[0], '--profile', 'full', *args[1:]]


if __name__ == '__main__':
    try:
        raise SystemExit(main(arguments(sys.argv[1:])))
    except ValueError as error:
        print(error, file=sys.stderr)
        raise SystemExit(2)
