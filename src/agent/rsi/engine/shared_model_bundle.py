"""Host admission and projection for shared-program model development.

One batch shares an execution context, not fitted parameters or acceptance.
The source tasks retain independently bound training/development/final data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from agent.rsi import model_tasks
from .context_feedback_dispatch import FIELD as FEEDBACK_FIELD
from .context_feedback_dispatch import policy_for
from .development_handoff import FIELD as HANDOFF_FIELD
from .development_handoff import POLICY as HANDOFF_POLICY
from .protein_model import digest
from agent.rsi.datasets.research_project_catalog import scope_key
from .shared_model_display import FIELD as DISPLAY_FIELD
from .shared_model_display import POLICIES as DISPLAY_POLICIES
from .shared_model_display import POLICY as DISPLAY_POLICY
from .storage import atomic_json, sha256_file
from .versions import VersionStore

SCHEMA = "shared_model_program.v1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def admit(tasks):
    """Reject source, duplicate-scope and cross-target partition errors before writing."""
    require(isinstance(tasks, list) and 1 <= len(tasks) <= 32, "Shared batch requires 1..32 tasks")
    data = {task["id"]: model_tasks.load_task(task) for task in tasks}
    require(len(data) == len(tasks), "Duplicate shared task identity")
    require(len({task["plugin_id"] for task in tasks}) == 1, "Shared programs require one model interface")
    require(len({task["paper"]["doi"] for task in tasks}) == 1, "A shared batch requires a common source study")
    scopes = set()
    phases = {name: set() for name in model_tasks.PHASES}
    contextual = tasks[0]["plugin_id"] in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }
    assay = tasks[0]["plugin_id"] in {"generated_assay.v1", "generated_antibody.v1"}
    antibody = tasks[0]["plugin_id"] == "generated_antibody.v1"
    entity_sequences = {}
    if assay:
        from .context_interfaces import module_for

        partition_identities = module_for(tasks[0]["plugin_id"]).partition_identities

        require(
            len({task["scope"]["grouping"]["level"] for task in tasks}) == 1,
            "Shared assay tasks require the same independence definition",
        )
    for task in tasks:
        scope = task["scope"]
        candidate = {
            "protein_scope": {"kind": "protein", "identifier": scope["protein"]},
            "property": scope["target"],
            "scenario": scope.get("context", {"assay": scope["assay"]}),
        }
        key = scope_key(candidate, task)
        require(key not in scopes, "Duplicate scientific scope in shared batch")
        scopes.add(key)
        for phase, rows in data[task["id"]].items():
            if assay:
                units = {identity for row in rows for identity in partition_identities(row, scope)}
                for row in rows:
                    entity = row["antibody"] if antibody else row["protein"]
                    sequence = (row["heavy"], row["light"]) if antibody else row["sequence"]
                    require(
                        entity not in entity_sequences or entity_sequences[entity] == sequence,
                        "Shared biological identity maps to inconsistent sequences",
                    )
                    entity_sequences[entity] = sequence
            else:
                units = {row["group"] if contextual else row["sequence"] for row in rows}
            require(
                not any(units & phases[p] for p in phases if p != phase),
                "Cross-task partition conflict in shared batch",
            )
            phases[phase].update(units)
    require(
        sum(Path(task["data"]["training"]["path"]).stat().st_size for task in tasks) <= 64 * 1024**2,
        "Shared training payload exceeds 64 MiB",
    )
    if antibody:
        from .generated_antibody import require_disjoint_parts

        combined = {phase: [row for task in tasks for row in data[task["id"]][phase]] for phase in model_tasks.PHASES}
        require_disjoint_parts(combined, tasks[0]["scope"])
    return data


def prepare_bundle(
    folder,
    workspace,
    tasks,
    identity,
    *,
    template_factory,
    phase="develop",
    registrations=None,
    training_baseline_feedback="none",
    tool_presentation=DISPLAY_POLICY,
):
    """Build the real native-MCP bundle from the existing single-model template.

    template_factory is run_paired_generated.make_bundle. Its Python runtime is
    copied once. Per-target folders contain declarations and accepted packets,
    never controller labels. No LLM call or library publication occurs here.
    """
    require(tool_presentation is None or tool_presentation in DISPLAY_POLICIES, "Unsupported shared tool presentation")
    require(phase in {"develop", "reuse"}, "Unknown shared-program phase")
    folder, workspace = Path(folder).absolute(), Path(workspace).absolute()
    require(
        folder != workspace and folder not in workspace.parents and workspace not in folder.parents,
        "Capability and workspace paths must not overlap",
    )
    for path in [folder, workspace]:
        require(not any(p.is_symlink() for p in [path, *path.parents]), "Shared paths cannot traverse symlinks")
        require(not path.exists(), "Shared output requires a fresh directory")
    data = admit(tasks)
    feedback = {}
    for task in tasks:
        feedback[task["id"]] = policy_for(
            {FEEDBACK_FIELD: training_baseline_feedback, "scope": task["scope"]}, plugin_id=task["plugin_id"]
        )
    if phase == "reuse":
        require(training_baseline_feedback == "none", "Reuse cannot receive training feedback")
        require(
            isinstance(registrations, dict) and set(registrations) == {t["id"] for t in tasks},
            "Reuse must cover every allocated target",
        )
    else:
        require(registrations is None, "Development cannot expose accepted model packets")
    first = tasks[0]
    inputs_name = "training.json" if phase == "develop" else "inputs.json"

    def projected(task):
        rows = data[task["id"]]
        return (
            rows["training"] if phase == "develop" else model_tasks.inference_inputs(rows["future"], task["plugin_id"])
        )

    kwargs = {}
    if phase == "reuse":
        kwargs.update(
            lifecycle=registrations[first["id"]]["lifecycle"], model_id=registrations[first["id"]]["model_id"]
        )
    _, _, template = template_factory(
        folder / "single_template",
        phase,
        {inputs_name: projected(first)},
        first["scope"],
        identity,
        plugin_id=first["plugin_id"],
        training_baseline_feedback=training_baseline_feedback,
        **kwargs,
    )
    source = folder / "shared_source"
    source.mkdir()
    for path in sorted(template.glob("*.py")):
        require(not path.is_symlink(), "Symlinked template implementation")
        name = "single_model_tools.py" if path.name == "generated_tools.py" else path.name
        shutil.copy2(path, source / name)
    shutil.copy2(Path(__file__).with_name("shared_model_tools.py"), source / "generated_tools.py")
    for name in ["shared_model_display.py", "training_feedback_view.py"]:
        shutil.copy2(Path(__file__).with_name(name), source / name)
    implementation = {p.name: sha256_file(p) for p in sorted(source.glob("*.py"))}
    workspace.mkdir(parents=True)
    targets = []
    for task in tasks:
        task_id = task["id"]
        target_source = source / "targets" / task_id
        target_source.mkdir(parents=True)
        target_work = workspace / "targets" / task_id
        target_work.mkdir(parents=True)
        rows = projected(task)
        child = {
            "phase": phase,
            "scope": task["scope"],
            "plugin_id": task["plugin_id"],
            "inputs": {inputs_name: digest(rows)},
            "shared_implementation_sha256": digest(implementation),
        }
        if feedback[task_id] != "none":
            child[FEEDBACK_FIELD] = feedback[task_id]
        if phase == "reuse":
            entry = registrations[task_id]
            packet = entry["lifecycle"].export(entry["model_id"], target_source / "registered")
            require(
                packet["model"]["scope"] == task["scope"] and packet["model"]["plugin_id"] == task["plugin_id"],
                "Accepted target scope mismatch",
            )
            require(
                packet["model"]["training_sha256"] == digest(data[task_id]["training"]),
                "Accepted target training data mismatch",
            )
            child["capability_sha256"] = digest(packet)
        atomic_json(target_source / "task.json", child)
        atomic_json(target_work / inputs_name, rows)
        targets.append({"id": task_id, "task_sha256": digest(child)})
    public = {
        "schema": SCHEMA,
        "phase": phase,
        "batch_id": digest({"phase": phase, "targets": targets}),
        "targets": targets,
        "implementation": implementation,
    }
    if tool_presentation is not None:
        public[DISPLAY_FIELD] = tool_presentation
    if phase == "develop":
        public[HANDOFF_FIELD] = HANDOFF_POLICY
    atomic_json(source / "task.json", public)
    admission = {
        "schema": SCHEMA,
        "batch_id": public["batch_id"],
        "tasks": [
            {
                "id": t["id"],
                "admission": model_tasks.admission(t),
                "data_hashes": {phase: digest(rows) for phase, rows in data[t["id"]].items()},
            }
            for t in tasks
        ],
        "shared_execution_count": 1,
        "allocated_scopes": len(tasks),
        "external_model_calls": 0,
        "new_autonomous_models": 0,
    }
    atomic_json(folder / "shared_admission.json", admission)
    versions = VersionStore(folder / "bundles")
    snapshot = versions.snapshot(
        source, metadata={"phase": phase, "model": identity, "kind": SCHEMA, "batch_id": public["batch_id"]}
    )
    return versions, snapshot["version_id"], folder / "bundles/versions" / snapshot["version_id"]
