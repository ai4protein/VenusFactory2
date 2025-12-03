import os
import subprocess
import sys
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--env', type=str, default='venv', help="venv, system")


args, remaining_args = parser.parse_known_args()

base_script = '.install/install'

if args.env == 'system':
    script_name = f"{base_script}_system.py"
else:
    script_name = f"{base_script}.py"

print(f"Language: {lang}")
print(f"Using script: {script_name}")

subprocess.run([sys.executable, script_name] + remaining_args)