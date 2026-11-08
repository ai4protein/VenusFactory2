from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from urllib.parse import urlparse
from pathlib import Path

from .checks import doctor, factory_files, factory_health, failed
from .common import InstallLock, Runner, atomic_json, inside, read_json
from .runtimes import Runtimes
from .system import Environment, choose_manager, frontend, probe
from . import harnesses, progress, components, skills, backbones as backbone_assets

ROOT = Path(__file__).resolve().parents[2]
STAGES = ("environment", "builtins", "frontend", "harnesses", "tools", "factory", "rsi", "skills")


def parser():
    p = argparse.ArgumentParser(description="Inspect, initialize and audit a VenusFactory2 checkout without root privileges")
    p.add_argument("command", nargs="?", choices=("plan", "install", "doctor"), default="plan")
    p.add_argument("--profile", choices=("minimal", "standard", "extended", "full"), default="minimal")
    p.add_argument('--components', help='Optional families: all, none, or external,factory,rsi,harnesses; platform and built-ins are always included')
    p.add_argument('--rsi-library', help='Attach an existing ScientistModelLibrary; publication/review checks remain mandatory')
    p.add_argument("--harnesses", help="Optional clients: kimi-code,deepseek-harness,codex,claude-code or all")
    p.add_argument("--dependencies", choices=("base", "platform"), help="base: Python only; platform: full API/MCP/ML dependencies")
    p.add_argument("--tools", help="Comma-separated tool IDs; replaces the profile selection. Use 'all' to include manual/GPU tools")
    p.add_argument("--alphafold-databases", choices=("none", "reduced", "full"), default="none",
                   help="Explicit AlphaFold2 database download: about 650/3000 GiB free required; default none")
    p.add_argument("--manager", choices=("auto", "uv", "conda"), default="auto")
    p.add_argument("--torch-type", choices=("auto", "cpu", "cu128"), default="auto")
    p.add_argument("--environment", default=".venv", help="Platform environment prefix, relative to the repository by default")
    p.add_argument("--use-existing-environment", action="store_true", help="Read-only reuse; do not install/update platform packages")
    p.add_argument("--stages", help="Only run the named stages; component flags infer their own stage when omitted")
    p.add_argument("--json", action="store_true", help="Machine-readable JSON on stdout; progress remains on stderr")
    p.add_argument("--no-progress", action="store_true", help="Suppress progress for scripts/CI")
    p.add_argument("--factory-preset", choices=("none", "demo", "predict-core", "predict-all", "proteinmpnn", "all"), default=None)
    p.add_argument("--backbone", action="append", default=[], help="HF foundation model to prefetch into the Factory runtime cache (repeatable)")
    p.add_argument("--verify-hashes", action="store_true", help="Hash installed assets during doctor instead of checking sizes only")
    p.add_argument("--hf-endpoint", help="Explicit HTTPS Hugging Face endpoint/mirror for public model downloads; default: official endpoint")
    p.add_argument("--service-url", action="append", default=[], help="Optional API runtime-config or MCP health URL; no services are started")
    p.add_argument("--report", help="Save a JSON plan/health report; install always saves reports on the host")
    return p


def selection(args, manifest):
    implicit = []
    if args.tools is not None:
        implicit.append('tools')
    if args.harnesses is not None:
        implicit.append('harnesses')
    if args.factory_preset is not None or args.backbone:
        implicit.append('factory')
    if args.rsi_library is not None:
        implicit.append('rsi')
    if args.profile == 'full' or args.components is not None:
        selected_families = components.families(args.components)
        default = ['environment', 'builtins', 'frontend', 'skills']
        default += [{'external':'tools'}.get(name, name) for name in selected_families]
        flags = {'external': args.tools, 'factory': args.factory_preset or args.backbone or None,
                 'rsi': args.rsi_library, 'harnesses': args.harnesses}
        for name,value in flags.items():
            if value is not None and name not in selected_families:
                raise ValueError(f'{name} options conflict with --components selection')
    else:
        default = implicit or (['environment'] if args.profile == 'minimal' else list(STAGES))
        if 'environment' in default and args.dependencies == 'platform' and 'builtins' not in default:
            default.append('builtins')
    stages = [x.strip() for x in args.stages.split(',') if x.strip()] if args.stages is not None else default
    if not stages:
        raise ValueError('Select at least one installation stage')
    # Built-in query/search support travels with a runnable platform, including
    # when callers explicitly select environment,frontend as in older commands.
    platform_dependencies = args.dependencies == 'platform' or (args.dependencies is None and (args.profile != 'minimal' or args.components is not None))
    if 'environment' in stages and platform_dependencies and 'builtins' not in stages:
        stages.append('builtins')
    if set(stages) - set(STAGES):
        raise ValueError("Unknown stage: " + ",".join(sorted(set(stages) - set(STAGES))))
    tool_profile = 'full' if args.components is not None and args.profile == 'minimal' else args.profile
    selected = list(manifest["tools"]) if args.tools == "all" else ([x.strip() for x in args.tools.split(",") if x.strip()] if args.tools is not None else manifest["profiles"][tool_profile])
    aliases = {"alphafold": "alphafold2", "proteinix": "protenix", "mol-1": "mol1", "mol_1": "mol1"}
    selected = [aliases.get(name.lower(), name.lower()) for name in selected]
    if args.alphafold_databases != "none" and ("tools" not in stages or "alphafold2" not in selected):
        raise ValueError("--alphafold-databases requires --tools alphafold2")
    if set(selected) - set(manifest["tools"]):
        raise ValueError("Unknown tool: " + ",".join(sorted(set(selected) - set(manifest["tools"]))))
    return [stage for stage in STAGES if stage in stages], list(dict.fromkeys(selected))


def install_factory(root, python, run, runtimes, preset, backbones):
    if preset == "none" and not backbones:
        return {"status": "skipped"}
    current = factory_health(root, preset, True)
    if current["corrupt"]:
        raise ValueError("Existing Factory files failed SHA256 verification; move corrupt files aside before retrying")
    if current["missing"]:
        # Explicit paths ignore unrelated shell overrides. The bundled manifest is authoritative.
        run([python, root / "scripts/download_ckpts.py", "--preset", preset,
             "--repo-id", "AI4Protein/VenusFactory2-ckpts", "--local-dir", root / "ckpt"])
    checked = factory_health(root, preset, True)
    if checked["missing"]:
        raise ValueError("Factory download returned without required files. Check the download log and network, or explicitly choose --hf-endpoint")
    if checked["status"] != "ready":
        raise ValueError("Downloaded Factory artifacts do not match ckpt/manifest.json")
    ids = backbones or (["ElnaggarLab/ankh-large", "facebook/esm2_t6_8M_UR50D"] if preset == "predict-core" else [])
    if ids:
        config = read_json(root / "configs/model_runtimes.json", {}).get("finetuned", {})
        cache = root / config.get("cache", "runtime/local/model-runtimes/factory/cache")
        inside(root, cache)
        for index, model in enumerate(ids, 1):
            progress.ui.line(f'  Factory backbone {index}/{len(ids)}: {model}')
            run([python, root / 'scripts/bootstrap/backbones.py', model, cache / 'huggingface/hub'])
        # The downloader environment cannot execute Factory models. Only configure
        # the cache here; inference uses the platform's separately installed runtime.
        runtimes.merge("finetuned", {"cache": runtimes.relative(cache)})
    checked["backbones_prefetched"] = ids
    if preset in {"predict-all", "all"} and not backbones:
        checked["backbone_note"] = "All task heads downloaded; select additional foundation models with --backbone according to their plm_model configuration"
    return checked


def main(argv=None):
    args = parser().parse_args(argv)
    progress.configure(not args.no_progress)
    root = ROOT
    try:
        manifest = read_json(root / "configs/bootstrap.json")
        endpoint = args.hf_endpoint or os.getenv("HF_ENDPOINT", "https://huggingface.co")
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in ("", "/"):
            raise ValueError("HF endpoint must be an HTTPS origin without credentials or query parameters")
        if args.hf_endpoint:
            os.environ["HF_ENDPOINT"] = endpoint.rstrip("/")
        stages, selected = selection(args, manifest)
        selected_harnesses = harnesses.names(args.harnesses if args.harnesses is not None else ('all' if args.profile == 'full' or args.components is not None else None)) if 'harnesses' in stages else []
        dependencies = args.dependencies or ('base' if args.profile == 'minimal' and args.components is None else 'platform')
        if dependencies == 'base' and (args.profile == 'full' or args.components is not None):
            raise ValueError('A full/component deployment needs platform dependencies; use initialize.py install for a base-only environment')
        preset = args.factory_preset or ('predict-all' if args.profile == 'full' else ('none' if args.profile == 'minimal' and args.components is None else 'predict-core'))
        backbones = args.backbone or (components.factory_backbones(root, manifest, preset) if args.profile == 'full' and 'factory' in stages else (['ElnaggarLab/ankh-large', 'facebook/esm2_t6_8M_UR50D'] if preset == 'predict-core' else []))
        system = probe(root)
        torch_type = system["torch_type"] if args.torch_type == "auto" else args.torch_type
        environment = Path(os.path.abspath(root / args.environment))
        python = environment / "bin/python"
        plan = {"system": system, "manager": choose_manager(args.manager), "dependencies": dependencies,
                "torch_type": torch_type, "environment": str(environment), "reuse_environment": args.use_existing_environment, "hf_endpoint": endpoint,
                "alphafold_databases": args.alphafold_databases,
                "stages": stages, "tools": [{"id": name, **manifest["tools"][name]} for name in selected] if "tools" in stages else [],
                "harnesses": selected_harnesses, "factory_preset": preset, "factory_bytes": sum(f["size"] for f in factory_files(root, preset)) if "factory" in stages else 0,
                "backbones": backbones,
                "builtins": components.BUILTINS if 'builtins' in stages else {},
                "component_families": [name for name,stage in [('external','tools'),('factory','factory'),('rsi','rsi'),('harnesses','harnesses')] if stage in stages],
                "note": "Download sizes exclude Python packages and foundation weights. Base prepares Python only; select platform dependencies to run API/MCP."}
        if args.command != 'doctor' and not args.json:
            print_plan(plan)
        if args.command == "plan":
            result = plan
        elif args.command == "doctor":
            result = doctor(root, python, selected, preset, system, hashes=args.verify_hashes, urls=args.service_url, stages=stages, dependencies=dependencies)
            if 'harnesses' in stages:
                result['harnesses'] = {name: harnesses.health(root, name) for name in selected_harnesses}
            if 'builtins' in stages:
                result['builtins'] = components.builtin_health(root, python)
            if 'rsi' in stages:
                result['rsi'] = components.rsi_health(root, python, args.rsi_library)
            if 'factory' in stages and backbones:
                result['backbones'] = backbone_assets.health(root, backbones, args.verify_hashes)
        else:
            if platform.system() != "Linux":
                raise ValueError("Full provisioning targets Linux; use a Linux host or WSL2. plan/doctor are read-only on other systems")
            if set(stages) & {'tools', 'skills'} and not system["executables"]["git"]:
                raise ValueError("Install Git with your OS package manager first")
            minimum = 0.5 + plan['factory_bytes'] / 2**30
            if 'environment' in stages:
                minimum += 12 if dependencies == 'platform' else 0.5
            if 'frontend' in stages:
                minimum += 2
            if 'tools' in stages and selected:
                minimum += 10
            if 'harnesses' in stages and selected_harnesses:
                minimum += 1
            if system['free_gib'] < minimum:
                raise ValueError(f'Insufficient disk space for selected components; need at least {minimum:.1f} GiB (large models may need more)')
            if not args.use_existing_environment:
                inside(root, environment)
            base = inside(root, "runtime/local/bootstrap")
            with InstallLock(base / "initialize.lock"):
                session = base / (time.strftime("%Y%m%d-%H%M%S") + f'-{time.time_ns() % 1000000000:09d}')
                session.mkdir(parents=True, exist_ok=True)
                atomic_json(session / "plan.json", plan)
                run = Runner(root, session / "install.log")
                env = Environment(root, run, args.manager, torch_type)
                runtimes = Runtimes(root, env, python, manifest, system["gpu"])
                runtimes.alphafold_databases = args.alphafold_databases
                result = {"steps": {}, "log": str(run.log), "plan": str(session / "plan.json")}
                for index, stage in enumerate(stages, 1):
                    progress.ui.line(f'[{index}/{len(stages)}] {stage}')
                    try:
                        if stage == "environment":
                            if dependencies == 'platform' or args.use_existing_environment:
                                python = env.platform(environment, args.use_existing_environment)
                            else:
                                python = env.create(environment, '3.12')
                            runtimes.python = python
                            status = {"status": "ready", "reused": args.use_existing_environment}
                        elif stage == "frontend":
                            frontend(root, run, python)
                            status = {"status": "ready"}
                        elif stage == 'builtins':
                            status = components.install_builtins(root, env, python)
                        elif stage == 'rsi':
                            status = components.install_rsi(root, python, args.rsi_library)
                        elif stage == 'harnesses':
                            status = {}
                            for i, name in enumerate(selected_harnesses, 1):
                                progress.ui.line(f'  Harness {i}/{len(selected_harnesses)}: {name}')
                                try:
                                    status[name] = harnesses.install(root, env, name)
                                except Exception as error:
                                    status[name] = {'status': 'failed', 'message': str(error)}
                                result['steps'][stage] = status
                                atomic_json(session / 'result.json', result)
                        elif stage == "tools":
                            status = {}
                            for name in selected:
                                progress.ui.line(f'  Tool {selected.index(name) + 1}/{len(selected)}: {name}')
                                try:
                                    status[name] = runtimes.install(name)
                                except Exception as error:
                                    status[name] = {"status": "failed", "message": str(error)}
                                progress.ui.line(f"  {name}: {status[name]['status']}" + (f" — {status[name]['message']}" if status[name].get("message") else ""))
                                result["steps"][stage] = status
                                atomic_json(session / "result.json", result)
                        elif stage == "factory":
                            factory_python = python
                            if preset != 'none' or backbones:
                                factory_python = env.create(base / 'factory-download', '3.12')
                                env.pip(factory_python, 'huggingface-hub==0.36.2', 'PyYAML>=6,<7')
                            status = install_factory(root, factory_python, run, runtimes, preset, backbones)
                        else:
                            status = skills.install(root, run, manifest["skills"])
                        result["steps"][stage] = status
                    except Exception as error:
                        result["steps"][stage] = {"status": "failed", "message": str(error)}
                        if stage == "environment":
                            break
                    atomic_json(session / "result.json", result)
                    progress.ui.line(f'[{index}/{len(stages)}] {stage}: {"needs attention" if failed(result["steps"][stage]) else "done"}')
                result["health"] = doctor(root, python, selected, preset, system, hashes=args.verify_hashes, urls=args.service_url, stages=stages, dependencies=dependencies)
                if 'harnesses' in stages:
                    result['health']['harnesses'] = {name: harnesses.health(root, name) for name in selected_harnesses}
                if 'builtins' in stages:
                    result['health']['builtins'] = components.builtin_health(root, python)
                if 'rsi' in stages:
                    result['health']['rsi'] = components.rsi_health(root, python)
                if 'factory' in stages and backbones:
                    result['health']['backbones'] = backbone_assets.health(root, backbones, args.verify_hashes)
                atomic_json(session / "result.json", result)
                atomic_json(base / "latest.json", {"report": str(session / "result.json")})
        if args.report:
            atomic_json(inside(root, args.report), result)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        elif args.command != 'plan':
            print_result(result)
        return 1 if args.command != "plan" and failed(result) else 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"Initialization stopped: {error}", file=sys.stderr)
        return 2


def print_plan(plan):
    system = plan['system']
    print(f"VenusFactory2 | {system['os']} {system['arch']} | {system['free_gib']} GiB free")
    print(f"Environment: {plan['manager']} / {plan['dependencies']} / {plan['torch_type']}")
    print('Stages: ' + ' -> '.join(plan['stages']))
    if plan['tools']:
        print('Tools: ' + ', '.join(row['id'] for row in plan['tools']))
        for row in plan['tools']:
            if row.get('manual'):
                print(f"  {row['id']} prerequisite: {row['manual']}")
            elif row.get('gpu') and not system.get('gpu'):
                print(f"  {row['id']}: NVIDIA GPU required; this host will report it as blocked")
    if plan['harnesses']:
        print('Harnesses: ' + ', '.join(plan['harnesses']))
    if plan.get('builtins'):
        print('Built in: databases, web search, public protein services, files, analysis and manual')
    if plan.get('component_families'):
        print('Optional families: ' + ', '.join(plan['component_families']))
    if 'rsi' in plan['stages']:
        print('RSI: attach the reviewed model library; an empty new library is valid')
    if 'factory' in plan['stages']:
        print(f"Factory: {plan['factory_preset']} ({progress.size(plan['factory_bytes'])} task weights; backbones additional)")
        if plan['backbones']:
            print('Factory backbones: ' + ', '.join(plan['backbones']))
    if plan['dependencies'] == 'base' and 'environment' in plan['stages']:
        print('Base installs Python only. Add --dependencies platform --stages environment,frontend to prepare the Web service.')
    if system['os'] != 'Linux':
        print('Install on Linux/WSL2. This host supports read-only planning and checks.')
    print(flush=True)


def print_result(result):
    print('Completed.' if not failed(result) else 'Some components need attention.')
    def visit(value, prefix=''):
        if not isinstance(value, dict):
            return
        if 'status' in value:
            print(f"  {prefix}: {value['status']}" + (f" — {value['message']}" if value.get('message') else ''))
        else:
            for key, item in value.items():
                if key not in ('system', 'plan'):
                    visit(item, prefix + '/' + key if prefix else key)
    visit(result)
    if result.get('log'):
        print('Log: ' + result['log'])
        print('Report: ' + str(Path(result['log']).with_name('result.json')))
        if failed(result):
            print('Retry the same command after resolving the reported issue; verified completed downloads are reused.')
