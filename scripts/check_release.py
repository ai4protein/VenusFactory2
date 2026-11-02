#!/usr/bin/env python3
"""Validate release metadata without importing application dependencies."""
import argparse
import json
from pathlib import Path
import re
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def check(root=ROOT, version=None):
    project = tomllib.loads((root / 'pyproject.toml').read_text())
    expected = version or project['project']['version']
    if not re.fullmatch(r'\d+\.\d+\.\d+', expected):
        return ['Use a stable semantic version: MAJOR.MINOR.PATCH']
    package = json.loads((root / 'frontend/package.json').read_text())
    lock = json.loads((root / 'frontend/package-lock.json').read_text())
    values = {'pyproject': project['project']['version'], 'frontend': package['version'],
              'lockfile': lock['version'], 'lockfile root': lock['packages']['']['version']}
    issues = [f'{name}: version does not match {expected}' for name, value in values.items() if value != expected]
    if not (root / f'docs/releases/v{expected}.md').is_file():
        issues.append('Release notes are missing')
    if (root / 'docker-compose.yaml').read_text().count('VENUS_FACTORY_VERSION:-'+expected) != 2:
        issues.append('Container image defaults must match the release')
    return issues


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version')
    args = parser.parse_args()
    issues = check(version=args.version)
    if issues:
        raise SystemExit('\n'.join(issues))
    print('Release versions and notes are consistent')
