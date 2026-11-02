"""Verified, frozen transfer of model implementations between Scientist goals.

Library data and fitted weights stay on the host. The bounded MCP projection
contains implementation precedents; every receiving goal must fit and qualify
its own model. Donor dependencies are verified in topological order.
"""

from __future__ import annotations

import ast
import copy
import json
import re
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from agent.harness.contracts import digest, file_hash

from agent.rsi import context_interfaces, context_transfer, model_tasks
from .experience import CONDITIONS, LIMITATION, validate_projection
from .model_origin_view import origin_view
from .storage import atomic_json, safe_child

MAX_DONORS = 5
_AUDIT = ContextVar("library_experience_audit", default=None)
TOOL = "read_model_library_experience"


class IneligibleDonor(ValueError):
    """A valid donor that is incompatible or overlaps the receiving holdouts."""


def require(value, message):
    if not value:
        raise ValueError(message)


def selection(value):
    require(
        isinstance(value, dict)
        and {"condition", "library_root", "entry_ids"}
        <= set(value)
        <= {"condition", "library_root", "entry_ids", "overlap_policy", "adaptation_policy"},
        "Invalid library experience selection",
    )
    require(value["condition"] in CONDITIONS, "Unknown library experience condition")
    context_transfer.validate_declaration(value)
    root = Path(value["library_root"])
    require(
        root.is_absolute() and ".." not in root.parts and not any(p.is_symlink() for p in [root, *root.parents]),
        "Library experience requires an absolute owner path",
    )
    ids = value["entry_ids"]
    require(
        isinstance(ids, list) and len(ids) <= MAX_DONORS and len(set(ids)) == len(ids),
        "Declare at most five distinct experience donors",
    )
    require(
        all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) for x in ids), "Invalid experience donor identity"
    )
    return copy.deepcopy(value)


@contextmanager
def audit_session():
    current = _AUDIT.get()
    if current is not None:
        yield current
        return
    token = _AUDIT.set({"verified": {}, "pending": set(), "dependencies": {}})
    try:
        yield _AUDIT.get()
    finally:
        _AUDIT.reset(token)


def read_frozen(root, state, protocol):
    if protocol.get("schema") == "shared_scientist_goal.v1":
        from .shared_library_experience import read_frozen as read_shared

        return read_shared(root, state, protocol)
    binding = protocol.get("library_experience")
    require(binding == state["definition"].get("library_experience"), "Goal library experience admission changed")
    if binding is None:
        return None
    require(set(binding) == {"manifest", "projection"}, "Invalid library experience binding")
    result = {}
    for name in ("manifest", "projection"):
        row = binding[name]
        require(set(row) == {"path", "sha256"}, "Invalid experience asset binding")
        require(row["path"] == f"library_experience/{name}.json", "Unexpected experience asset path")
        path = safe_child(root, row["path"])
        require(
            path.is_file() and not path.is_symlink() and file_hash(path) == row["sha256"],
            "Frozen library experience changed",
        )
        result[name] = json.loads(path.read_text())
    manifest = result["manifest"]
    extra = {"context_admission"} if "overlap_policy" in manifest.get("selection", {}) else set()
    require(
        set(manifest) == {"schema", "selection", "target_admission", "donors", "overlap_rule"} | extra
        and manifest["schema"] == 1,
        "Invalid experience manifest",
    )
    selection(manifest["selection"])
    validate_projection(result["projection"])
    return result


def verified_entries(library_root, identities):
    """Iterative dependency replay; the cache exists only during this audit."""
    from .scientist_model_library import ScientistModelLibrary

    with audit_session() as session:
        stack = [(str(library_root), identity, False) for identity in reversed(identities)]
        while stack:
            root, identity, expanded = stack.pop()
            key = (root, identity)
            if key in session["verified"]:
                continue
            library = ScientistModelLibrary(root)
            if expanded:
                # Dependencies are already replayed, so nested goal validation
                # reads the operation-local cache without recursive traversal.
                value = library.resolve(identity, replay_training=True)
                session["verified"][key] = value
                session["pending"].remove(key)
                continue
            require(key not in session["pending"], "Cyclic library experience ancestry")
            session["pending"].add(key)
            entry = library._object("entries", identity)
            origin = library._object("origins/" + digest(entry["key"]), entry["canonical_origin"])
            goal = library.goal(safe_child(library.owner, origin["goal"]))
            state = json.loads((goal / "goal.json").read_text())
            protocol = json.loads((goal / "protocol.json").read_text())
            frozen = read_frozen(goal, state, protocol)
            stack.append((root, identity, True))
            session["dependencies"][key] = []
            if frozen:
                request = frozen["manifest"]["selection"]
                session["dependencies"][key] = [(request["library_root"], child) for child in request["entry_ids"]]
                stack.extend((request["library_root"], child, False) for child in reversed(request["entry_ids"]))
        return [session["verified"][(str(library_root), identity)] for identity in identities]


def compatible(source_plugin, target_plugin, source_scope, target_scope, *, adaptation_policy=None):
    if adaptation_policy is not None:
        require(adaptation_policy == context_transfer.SOURCE_ADAPTATION, "Unsupported source adaptation policy")
        if source_plugin not in context_interfaces.PLUGINS or target_plugin not in context_interfaces.PLUGINS:
            return False
        # This admits a program precedent, not a fitted predictor or a mapping
        # between physical variables. The recipient's worker and assessment
        # enforce its own input contract after refitting or source revision.
        context_interfaces.module_for(source_plugin).validate_scope(source_scope)
        context_interfaces.module_for(target_plugin).validate_scope(target_scope)
        return True
    if source_plugin == target_plugin and source_plugin in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
    }:
        return context_transfer.interface(source_scope) == context_transfer.interface(target_scope)
    sequences = {"generated_python.v1", "generated_sequence.v1"}
    classifiers = {"generated_classifier.v1", "generated_classifier.large.v1"}
    if source_plugin in sequences and target_plugin in sequences:
        return True
    if source_plugin in classifiers and target_plugin in classifiers:
        return True
    return (
        source_plugin == target_plugin == "generated_features.v1"
        and source_scope["representation"] == target_scope["representation"]
    )


def describe_source(source):
    """Deterministic structural summary without literals, records or weights."""
    tree = ast.parse(source)
    aliases, imports, routines, calls = {}, set(), set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name
                imports.add(item.name)
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                name = (node.module or "") + "." + item.name
                aliases[item.asname or item.name] = name
                imports.add(name)
        elif isinstance(node, ast.FunctionDef):
            routines.add(node.name)

    def qualified(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            prefix = qualified(node.value)
            return prefix + "." + node.attr if prefix else None
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = qualified(node.func)
            if name:
                calls.add(name)

    def names(values, limit):
        return (
            ", ".join(sorted(value for value in values if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]{0,99}", value))[:limit])
            or "none statically identified"
        )

    return (
        "Source structure: imports "
        + names(imports, 8)
        + "; routines "
        + names(routines, 12)
        + "; numerical library calls "
        + names(calls, 24)
        + ". This is a structural description, not a mechanistic explanation. Refit and independently validate on the receiving task."
    )


def snapshot(request, target):
    request = selection(request)
    data = model_tasks.load_task(target)
    contextual = target["plugin_id"] in {"generated_context.v1", "generated_context.v2", "generated_context.v3"}
    policy = context_transfer.validate_declaration(request, plugin_id=target["plugin_id"])
    context_admission = context_transfer.describe_task(target, data) if policy else None
    if contextual and request["entry_ids"] and policy is None:
        raise IneligibleDonor(
            "Cross-task contextual source transfer requires a separately reviewed study/group overlap protocol"
        )
    protected = set() if contextual else {row["sequence"] for phase in ("validation", "future") for row in data[phase]}
    records, projected = [], []
    with audit_session() as session:
        verified = verified_entries(request["library_root"], request["entry_ids"])
        for identity, value in zip(request["entry_ids"], verified, strict=True):
            model = value["packet"]["model"]
            if not compatible(
                model["plugin_id"],
                target["plugin_id"],
                model["scope"],
                target["scope"],
                adaptation_policy=request.get("adaptation_policy"),
            ):
                raise IneligibleDonor("Incompatible donor model interface")
            origin = Path(value["origin_goal"])
            source_view = origin_view(value)
            seen, ancestry, visited = set(), [], set()
            start = (request["library_root"], identity)
            pending = [start]
            while pending:
                key = pending.pop()
                if key in visited:
                    continue
                visited.add(key)
                ancestor = value if key == start else session["verified"][key]
                ancestor_view = origin_view(ancestor)
                cohorts = ancestor_view["data"]
                for phase in model_tasks.PHASES:
                    rows = cohorts[phase]
                    if not contextual:
                        seen.update(row["sequence"] for row in rows)
                context_record = {}
                if contextual:
                    ancestor_task = ancestor_view["task"]
                    admitted = model_tasks.load_task(ancestor_task)
                    require(admitted == cohorts, "Donor contextual task differs from its frozen cohorts")
                    require(
                        ancestor_task["scope"] == ancestor["packet"]["model"]["scope"],
                        "Donor context scope differs from its published model",
                    )
                    descriptor = context_transfer.describe_task(ancestor_task, cohorts)
                    if descriptor["study_doi"] == context_admission["study_doi"]:
                        raise IneligibleDonor("Donor ancestry contextual study overlaps the receiving study")
                    context_record["context_admission"] = descriptor
                ancestry.append(
                    {
                        "library_root": key[0],
                        "entry_id": key[1],
                        "origin_sha256": digest(ancestor["origin"]),
                        "data_hashes": ancestor_view["data_hashes"],
                        **context_record,
                    }
                )
                pending.extend(session["dependencies"].get(key, []))
            if protected & seen:
                raise IneligibleDonor("Donor ancestry cohorts overlap target validation or future sequences")
            code = model["recipe"]["source_code"]
            record = {
                "entry_id": identity,
                "key_id": value["key_id"],
                "origin_goal": str(origin),
                "origin_sha256": digest(value["origin"]),
                "packet_sha256": digest(value["packet"]),
                "source_sha256": model["source_sha256"],
                "data_hashes": source_view["data_hashes"],
                "ancestry": sorted(ancestry, key=lambda x: (x["library_root"], x["entry_id"])),
                "overlap_count": 0,
            }
            records.append(record)
            if request["condition"] != "none":
                summary = describe_source(code)
                if "adaptation_policy" in request:
                    summary = (
                        "Numeric source adaptation precedent: the donor has "
                        + str(len(model["scope"]["covariates"]))
                        + " input columns; this receiving task has "
                        + str(len(target["scope"]["covariates"]))
                        + ". Variable meanings, endpoints and units may differ. Inspect the receiving task, "
                        "adapt the implementation when needed, and fit new parameters. "
                        "This supplies a modelling method, not transferred biological predictions. " + summary
                    )
                item = {
                    "experience_id": digest(record),
                    "source_sha256": model["source_sha256"],
                    "donor_scope": model["scope"],
                    "method_summary": summary,
                    "acceptance": {"passed": True, "training_replayed": True},
                    "parents": [],
                }
                if request["condition"] == "source":
                    item["source_code"] = code
                projected.append(item)
    projection = validate_projection(
        {"schema": 1, "condition": request["condition"], "entries": projected, "limitation": LIMITATION}
    )
    require(len(json.dumps(projection).encode()) <= 200_000, "Experience projection exceeds its bounded context budget")
    return {
        "manifest": {
            "schema": 1,
            "selection": request,
            "target_admission": model_tasks.admission(target),
            "donors": records,
            "overlap_rule": context_transfer.RULE
            if policy
            else "All direct and ancestral donor cohorts must be exactly sequence-disjoint from target validation and future; no homology-independence claim",
            **({"context_admission": context_admission} if policy else {}),
        },
        "projection": projection,
    }


def freeze(root, value):
    folder = root / "library_experience"
    folder.mkdir(exist_ok=False)
    binding = {}
    for name in ("manifest", "projection"):
        path = folder / (name + ".json")
        atomic_json(path, value[name])
        binding[name] = {"path": str(path.relative_to(root)), "sha256": file_hash(path)}
    return binding


def verify_goal(root, state, protocol):
    frozen = read_frozen(root, state, protocol)
    if frozen is None:
        return None
    target = json.loads((root / "model_task.json").read_text())
    current = snapshot(frozen["manifest"]["selection"], target)
    require(current == frozen, "Library experience provenance changed")
    return current["projection"]


def verify_episode(root, state, protocol, cycle):
    if protocol.get("schema") == "shared_scientist_goal.v1":
        from .shared_library_experience import verify_episode as verify_shared

        return verify_shared(root, state, protocol, cycle)
    frozen = read_frozen(root, state, protocol)
    if frozen is None:
        return
    folder = safe_child(root, f"episodes/cycle_{cycle['index']:03d}")
    versions = folder / "capability/bundles/versions"
    if not versions.is_dir():
        require(cycle["status"] != "completed", "Completed episode has no experience capability")
        return
    caps = list(versions.iterdir())
    require(len(caps) == 1, "Ambiguous experience capability")
    task = json.loads((caps[0] / "task.json").read_text())
    lock = json.loads((folder / "execution/profile.lock.json").read_text())
    enabled = cycle.get("phase") == "develop"
    if not enabled:
        require(
            "library_experience_sha256" not in task and TOOL not in lock["task"]["tools"],
            "Library experience leaked into model reuse",
        )
        return
    require(
        task.get("library_experience_sha256") == digest(frozen["projection"]),
        "Episode library experience selection changed",
    )
    require(
        json.loads((caps[0] / "library_experience.json").read_text()) == frozen["projection"],
        "Episode experience content changed",
    )
    require(TOOL in lock["task"]["tools"], "Episode lacks declared library experience tool")
    if cycle["status"] == "completed":
        events = [json.loads(line) for line in (folder / "execution/events.jsonl").read_text().splitlines()]
        calls = {
            row["tool_call_id"]
            for row in events
            if row.get("kind") == "tool_call_start" and row.get("tool_name") == "mcp__venusfactory__" + TOOL
        }
        require(
            any(
                row.get("kind") == "tool_result" and row.get("tool_call_id") in calls and not row.get("is_error", False)
                for row in events
            ),
            "Completed development episode did not read library experience",
        )


def campaign_policy(value):
    require(isinstance(value, dict) and value.get("condition") in CONDITIONS, "Invalid campaign experience condition")
    context_transfer.validate_declaration(value)
    extra = {key for key in ("overlap_policy", "adaptation_policy") if key in value}
    if value.get("selection") == "explicit":
        require(
            set(value) == {"condition", "selection", "entry_ids"} | extra, "Invalid explicit campaign experience fields"
        )
        ids = value["entry_ids"]
        require(
            isinstance(ids, list) and len(ids) <= MAX_DONORS and len(set(ids)) == len(ids),
            "Invalid explicit experience donors",
        )
        require(
            all(isinstance(x, str) and re.fullmatch(r"[0-9a-f]{64}", x) for x in ids), "Invalid explicit donor identity"
        )
    else:
        require(
            set(value) == {"condition", "selection", "max_entries"} | extra and value["selection"] == "prior_campaign",
            "Unknown campaign experience policy",
        )
        require(
            type(value["max_entries"]) is int and 1 <= value["max_entries"] <= MAX_DONORS,
            "Invalid experience retrieval limit",
        )
    return copy.deepcopy(value)


def select_for_campaign(policy, library_root, candidates, target):
    """First eligible prior publications; never rank using target performance."""
    policy = campaign_policy(policy)
    context_transfer.validate_declaration(policy, plugin_id=target["plugin_id"])
    extra = {key: policy[key] for key in ("overlap_policy", "adaptation_policy") if key in policy}
    if policy["selection"] == "explicit":
        return {
            "condition": policy["condition"],
            "library_root": str(library_root),
            "entry_ids": policy["entry_ids"],
            **extra,
        }, []
    selected, review = [], []
    with audit_session():
        for identity in dict.fromkeys(candidates):
            request = {"condition": "none", "library_root": str(library_root), "entry_ids": [identity], **extra}
            try:
                snapshot(request, target)
            except IneligibleDonor as error:
                review.append({"entry_id": identity, "status": "ineligible", "reason": str(error)})
                continue
            selected.append(identity)
            review.append({"entry_id": identity, "status": "selected"})
            if len(selected) == policy["max_entries"]:
                break
    return {"condition": policy["condition"], "library_root": str(library_root), "entry_ids": selected, **extra}, review
