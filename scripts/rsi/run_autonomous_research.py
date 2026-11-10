#!/usr/bin/env python3
"""Detached, resumable natural-language Scientist goal runner."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from agent.harness import Profile
from agent.harness.catalog import resolve_credential
from agent.harness.installation import default_runtimes
from agent.harness.autonomous_research import AutonomousResearch, create
from agent.science_goal import GoalStore

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['create','run','resume']);p.add_argument('--output',required=True);p.add_argument('--profile');p.add_argument('--objective');p.add_argument('--credential-toml')
    for name,default in [('max-cycles',8),('model-calls',192),('tool-calls',96),('wall-seconds',7200)]:p.add_argument('--'+name,type=int,default=default)
    args=p.parse_args()
    if args.action=='create':
        create(args.output,Profile.from_dict(json.loads(Path(args.profile).read_text())),args.objective,{k:getattr(args,k) for k in ['max_cycles','model_calls','tool_calls','wall_seconds']});return
    store=GoalStore(args.output);profile=Profile.from_dict(store.read()['definition']['profile'])
    if args.action=='resume':store.resume()
    runtime=AutonomousResearch(store,resolve_credential(profile.model,args.credential_toml),default_runtimes(ROOT))
    result=asyncio.run(runtime.run());print(json.dumps({'status':result['status'],'cycles':len(result['cycles'])}))

if __name__=='__main__':main()
