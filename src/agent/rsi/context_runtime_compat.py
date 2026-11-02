"""Read-only compatibility for additive context-group metadata declarations.

All numerical code and dependencies must be unchanged. Archived controller code
is parsed, never executed. Callers must replay training, validation and inference.
New assessments still require their exact frozen plugin implementation hash.
"""

import ast
import hashlib
import json
from pathlib import Path

from .numerical_evidence import verify_assessment
from .protein_model import digest
from .storage import safe_child, sha256_file
from .versions import VersionStore

FILES = (
    "generated_context.py",
    "code_sandbox.py",
    "code_worker.py",
    "protein_references.py",
    "numeric_runtime.py",
    "protein_model.py",
)


def metadata_extension(archived, current):
    def extract(source):
        tree = ast.parse(source)
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "GROUP_LEVELS" for t in node.targets)
        ]
        if len(assignments) != 1 or len(assignments[0].targets) != 1:
            raise ValueError("Ambiguous context group declaration")
        levels = ast.literal_eval(assignments[0].value)
        if not isinstance(levels, set) or not levels or any(type(v) is not str for v in levels):
            raise ValueError("Literal context group metadata required")
        tree.body.remove(assignments[0])
        return levels, ast.dump(tree, include_attributes=False)

    old, old_ast = extract(archived)
    new, new_ast = extract(current)
    if not old < new or old_ast != new_ast:
        raise ValueError("Context numerical code changed beyond an additive group declaration")
    return old


def verify_sources(root, policy, protocol, *, current_root=None):
    current_root = Path(current_root) if current_root else Path(__file__).resolve().parent
    sources = []
    for name in FILES:
        relative = "src/agent/rsi/" + name
        archived = safe_child(root, "implementation/" + relative)
        if sha256_file(archived) != protocol["implementation"][relative]:
            raise ValueError("Archived context implementation changed")
        content = archived.read_bytes()
        sources.append(content)
        current = (current_root / name).read_bytes()
        if name == "generated_context.py":
            levels = metadata_extension(content, current)
        elif content != current:
            raise ValueError("Context numerical dependency changed")
    if hashlib.sha256(b"".join(sources)).hexdigest() != policy["plugin_sha256"]:
        raise ValueError("Original context policy does not bind the archived implementation")
    if policy["scope"]["grouping"]["level"] not in levels:
        raise ValueError("Historical scope was not admitted by its frozen group declaration")


def read_compatible_context(root, controller, model_id, policy, protocol):
    if (
        set(policy)
        != {
            "schema",
            "scope",
            "plugin_id",
            "plugin_sha256",
            "training_sha256",
            "validation_sha256",
            "minimum_skill",
            "max_candidates",
        }
        or policy["schema"] != 1
        or policy["plugin_id"] != "generated_context.v1"
        or policy["max_candidates"] != 1
        or type(policy["minimum_skill"]) not in (float, int)
        or not 0 <= policy["minimum_skill"] < 1
    ):
        raise ValueError("Unsupported historical context policy")
    verify_sources(root, policy, protocol)
    manifest = VersionStore(controller / "artifacts").verify(model_id)
    state = json.loads((controller / "state.json").read_text())
    if (
        state.get("schema") != 1
        or not isinstance(state.get("candidates"), dict)
        or not isinstance(state.get("published"), list)
        or not isinstance(state.get("events"), list)
    ):
        raise ValueError("Invalid context lifecycle history")
    candidate = state["candidates"][model_id]
    if manifest["metadata"] != {
        "session_id": candidate["session_id"],
        "mode": candidate["mode"],
        "policy_sha256": digest(policy),
    }:
        raise ValueError("Context artifact policy binding changed")
    model = json.loads(safe_child(controller, f"artifacts/versions/{model_id}/model.json").read_text())
    report = json.loads(safe_child(controller, f"assessment_{model_id}.json").read_text())
    if (
        report.get("schema") != 1
        or report.get("model_id") != model_id
        or report.get("policy_sha256") != digest(policy)
        or report.get("plugin_sha256") != policy["plugin_sha256"]
        or report.get("passed") is not True
        or report.get("training_replayed") is not True
        or digest(report) != candidate["assessment_sha256"]
    ):
        raise ValueError("Context assessment binding changed")
    if verify_assessment(controller, model_id, model, policy, report)["status"] != "verified":
        raise ValueError("Per-row numerical evidence required for context compatibility")
    return state, model, report
