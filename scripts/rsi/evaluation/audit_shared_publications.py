"""Resume an offline publication audit without changing the originating campaign.

Runs the campaign's frozen validators and numerical budgets. A completed batch
can be reused only when its entire evidence tree, task inputs, library objects,
runtime source and this auditor still have the same hashes.
"""
from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def tree_hash(root):
    root = Path(root)
    if not root.is_dir() or any(p.is_symlink() for p in [root, *root.parents]):
        raise ValueError("Missing or symlinked evidence tree")
    hashes = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError("Symlinked evidence")
        if p.is_file() and "__pycache__" not in p.parts and not p.name.endswith(".lock"):
            hashes[p.relative_to(root).as_posix()] = file_hash(p)
    return digest(hashes)


def completed_checkpoint(path, binding):
    if not Path(path).exists():
        return None
    packet = read(path)
    if set(packet) != {"record", "sha256"} or digest(packet["record"]) != packet["sha256"]:
        raise ValueError("Audit checkpoint changed")
    record = packet["record"]
    if record["binding"] != binding:
        raise ValueError("Audit evidence changed; use a new audit output directory")
    return record if record["status"] in {"passed", "no_publication"} else None


def summary(records, expected):
    verified, sources, statuses = {}, set(), {}
    for record in records:
        statuses[record["status"]] = statuses.get(record["status"], 0) + 1
        if record["status"] != "passed":
            continue
        for row in record["verified"]:
            scope = row["scope_id"]
            if scope in verified:
                raise ValueError("Repeated scientific scope in audit")
            verified[scope] = row
            sources.add(row["program_sha256"])
    return {
        "schema": "shared_publication_reaudit.v1", "expected_batches": expected,
        "processed_batches": len(records), "batch_statuses": statuses,
        "finished": len(records) == expected,
        "all_publication_checks_passed": len(records) == expected and not statuses.get("failed", 0),
        "verified_scope_count": len(verified), "distinct_verified_programs": len(sources),
        "verified_scope_ids": sorted(verified), "external_model_calls": 0,
        "new_autonomous_models": 0, "final_labels_scored": False,
    }


def runtime_module(repository, area, name):
    """Load the original package layout without rewriting frozen source."""
    repository = Path(repository).resolve()
    base = repository / 'src/agent/rsi'
    if (base / area / (name + '.py')).is_file():
        qualified = 'agent.rsi.' + area + '.' + name
    elif (base / (name + '.py')).is_file():
        qualified = 'agent.rsi.' + name
    else:
        raise ValueError('Frozen runtime module is missing: ' + name)
    module = importlib.import_module(qualified)
    if not Path(module.__file__).resolve().is_relative_to(repository / 'src'):
        raise ValueError('Auditor imported a different runtime; use a fresh process')
    return module


def load_campaign(campaign):
    initial = read(campaign / "campaign.json")
    repository = Path(initial["definition"]["repository"]).resolve()
    sys.path.insert(0, str(repository / "src"))
    campaign_runtime = runtime_module(repository, 'campaigns', 'shared_model_campaign')
    SharedCampaignStore, TERMINAL = campaign_runtime.SharedCampaignStore, campaign_runtime.TERMINAL
    ScientistModelLibrary = runtime_module(repository, 'engine', 'scientist_model_library').ScientistModelLibrary
    store = SharedCampaignStore(campaign)
    state = store.read()
    if state["owner"] is not None or state["inflight"] is not None or any(b["status"] not in TERMINAL for b in state["batches"].values()):
        raise ValueError("Audit only a settled campaign without an active owner")
    library = ScientistModelLibrary(state["definition"]["library_root"])
    return store, state, store.specifications(state), library, repository


def batch_binding(store, state, spec, library, name):
    from agent.rsi import model_tasks
    goal = store.goal(name)
    tasks = {}
    for task in spec["tasks"]:
        tasks[task["id"]] = {"admission": model_tasks.admission(task),
                             "data": digest(model_tasks.load_task(task)),
                             "source_split": model_tasks.source_and_split(task)}
    reports = [a["publication"] for a in state["batches"][name]["attempts"] if a.get("publication")]
    objects = {}
    for report in reports:
        objects[report["report_id"]] = library._object("publication_attempts", report["report_id"])
        for row in report["outcomes"]:
            if row.get("status") != "published":
                continue
            entry = library._object("entries", row["entry_id"])
            key = row["key_id"]
            objects[row["entry_id"]] = entry
            objects["index/" + key] = read(library.root / "index" / (key + ".json"))
            objects[entry["packet_id"]] = library._object("packets", entry["packet_id"])
            objects[entry["canonical_origin"]] = library._object("origins/" + key, entry["canonical_origin"])
    return {"campaign_sha256": file_hash(store.root / "campaign.json"), "batch": name,
            "goal_tree_sha256": tree_hash(goal), "tasks_sha256": digest(tasks),
            "library_objects_sha256": digest(objects), "auditor_sha256": file_hash(__file__),
            "runtime_sha256": digest(state["definition"]["implementation"]),
            "python": sys.version, "executable": sys.executable}


def audit_batch(store, state, spec, library, name):
    from agent.harness.contracts import digest as native_digest
    repository = Path(state['definition']['repository'])
    campaign_runtime = runtime_module(repository, 'campaigns', 'shared_model_campaign')
    bind_goal, task_scope = campaign_runtime.bind_goal, campaign_runtime.task_scope
    shared_campaign_experience = campaign_runtime.shared_campaign_experience
    validate_shared_goal = runtime_module(repository, 'engine', 'shared_model_publication').validate_shared_goal
    reports = [a["publication"] for a in state["batches"][name]["attempts"] if a.get("publication")]
    if not reports:
        return {"status": "no_publication", "verified": [], "original_status": state["batches"][name]["status"]}
    if len(reports) != 1:
        raise ValueError("Repeated publication for a shared batch")
    report = reports[0]
    stored = library._object("publication_attempts", report["report_id"])
    if stored != {k: v for k, v in report.items() if k != "report_id"}:
        raise ValueError("Shared publication report changed")
    goal = store.goal(name)
    bind_goal(store, spec, goal)
    if shared_campaign_experience.FIELD in spec:
        shared_campaign_experience.verify_selection(store, spec, goal=goal, replay=True)
    if stored["kind"] != "shared_model_goal_publication_attempt" or stored["origin_goal"] != str(goal.relative_to(library.owner)):
        raise ValueError("Foreign publication")
    values = {v["origin"]["task_id"]: v for v in validate_shared_goal(goal, replay_training=True)}
    if set(values) != {v["task_id"] for v in stored["outcomes"]}:
        raise ValueError("Publication target coverage changed")
    tasks = {task["id"]: task for task in spec["tasks"]}
    verified = []
    for outcome in stored["outcomes"]:
        value = values[outcome["task_id"]]
        entry = library._object("entries", outcome["entry_id"])
        packet = library._object("packets", entry["packet_id"])
        key = native_digest(entry["key"])
        origin = library._object("origins/" + key, entry["canonical_origin"])
        if not (outcome["status"] == "published" and outcome["model_id"] == value["origin"]["model_id"]
                and entry["key"] == value["key"] and packet == value["packet"]
                and entry["packet_id"] == native_digest(value["packet"])
                and read(library.root / "index" / (key + ".json")) == {"entry_id": outcome["entry_id"]}
                and origin["goal"] == str(goal.relative_to(library.owner))
                and value["origin"] == {k: v for k, v in origin.items() if k not in {"goal", "key_id", "packet_id"}}):
            raise ValueError("Registered model, numerical replay or canonical origin differs")
        task = tasks[outcome["task_id"]]
        verified.append({"task_id": task["id"], "entry_id": outcome["entry_id"], "scope_id": task_scope(task),
                         "program_sha256": value["key"]["program_sha256"], "stage": task["stage"],
                         "skill": value["skill"], "reuse_receipt_sha256": value["reuse_receipt_sha256"]})
    return {"status": "passed", "verified": verified, "original_status": state["batches"][name]["status"]}


def run(campaign, output, limit=None):
    import fcntl
    campaign, output = Path(campaign).resolve(), Path(output).resolve()
    if output == campaign or campaign in output.parents:
        raise ValueError("Audit output must be separate from original campaign evidence")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".audit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        store, state, specs, library, repository = load_campaign(campaign)
        master = {"campaign": str(campaign), "campaign_sha256": file_hash(campaign / "campaign.json"),
                  "auditor_sha256": file_hash(__file__), "runtime": str(repository),
                  "runtime_sha256": digest(state["definition"]["implementation"])}
        if (output / "binding.json").exists() and read(output / "binding.json") != master:
            raise ValueError("Audit binding changed; use a new output directory")
        write(output / "binding.json", master)
        records = []
        for index, (name, spec) in enumerate(specs.items()):
            if limit is not None and index >= limit:
                break
            print(json.dumps({"batch": name, "phase": "binding"}), flush=True)
            binding = batch_binding(store, state, spec, library, name)
            checkpoint = output / "batches" / (name + ".json")
            record = completed_checkpoint(checkpoint, binding)
            if record is None:
                started = time.monotonic()
                write(output / "progress.json", {**summary(records, len(specs)), "current_batch": name, "phase": "numerical_replay"})
                try:
                    result = audit_batch(store, state, spec, library, name)
                except Exception as error:
                    result = {"status": "failed", "verified": [], "error_type": type(error).__name__, "error": str(error)}
                    folder = output / "failures"; folder.mkdir(exist_ok=True)
                    (folder / f"{name}-{time.time_ns()}.txt").write_text(traceback.format_exc())
                if batch_binding(store, state, spec, library, name) != binding:
                    raise ValueError("Evidence changed during audit")
                record = {"binding": binding, "elapsed_seconds": time.monotonic() - started, **result}
                packet = {"record": record, "sha256": digest(record)}
                write(output / "attempts" / f"{name}-{time.time_ns()}.json", packet)
                write(checkpoint, packet)
            records.append(record)
            report = {**summary(records, len(specs)), "current_batch": None, "original_campaign_status": state["status"]}
            write(output / "progress.json", report)
            print(json.dumps({"batch": name, "status": record["status"], "verified": len(record["verified"]), "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
        if file_hash(campaign / "campaign.json") != master["campaign_sha256"]:
            raise ValueError("Source campaign changed")
        store.read()  # Recheck frozen source bytes after numerical execution.
        report = {**summary(records, len(specs)), "original_campaign_status": state["status"], "source_campaign_unchanged": True}
        write(output / "summary.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-batches", type=int)
    args = parser.parse_args()
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error("--limit-batches must be positive")
    print(json.dumps(run(args.campaign, args.output, args.limit_batches), indent=2))


if __name__ == "__main__":
    main()
