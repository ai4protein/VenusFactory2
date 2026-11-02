#!/usr/bin/env python3
"""Install the optional gget runtime without changing the web environment."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / '.venvs' / 'public-query'
GGET_VERSION = '0.30.8'

def main():
    python = RUNTIME / 'bin' / 'python'
    if not python.exists():
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(RUNTIME)], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check', '--no-input',
                    '--index-url', 'https://pypi.org/simple', 'gget==' + GGET_VERSION], check=True)
    subprocess.run([str(python), '-c', 'import gget; print("gget", gget.__version__)'], check=True)

if __name__ == '__main__': main()
