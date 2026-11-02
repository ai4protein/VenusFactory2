"""MCP development tools for agent-written model source with training-only feedback."""

from __future__ import annotations

import ast
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path

import generated_model
from code_sandbox import CandidateCodeError, ModelOutputError, guard_source
from experience import LIMITATION, validate_projection
from fastmcp import FastMCP
from protein_model import digest
from protein_references import audit_source_references, reference_catalog

mcp = FastMCP("VenusFactoryGeneratedModels")
ROOT = Path(os.environ.get("VENUS_RSI_WORKSPACE", "/workspace")).resolve()
BUNDLE = Path(__file__).resolve().parent
DEVELOPMENT_LEDGER_SCHEMA = 1
LEDGER = ".model_development.json"


def authoring_contract(*, include_references):
    """Explain the existing ABI without supplying a fitted model or predictor."""
    result = {
        "schema": 1,
        "rules": [
            "Submit complete Python source, including every import and helper used by fit and predict.",
            "fit must return only JSON-compatible parameters; convert numpy arrays/scalars to lists/Python numbers. Do not return an estimator object.",
            "predict must reconstruct computation from those parameters, preserving input order and returning finite outputs in the inspected ABI.",
            "Fit feature transforms on training data only and store their parameters for predict; use the supplied seed for stochastic estimators.",
            "After failed feedback, edit the actual source before retrying. Changing only rationale does not repair code. Failed attempts consume the same two-attempt budget.",
        ],
    }
    if include_references:
        catalog = reference_catalog()
        examples = []
        for symbol, reference_id in [
            ("MASS", "free_amino_acid.average_mass.v1"),
            ("HYDROPATHY", "kyte_doolittle.hydropathy.v1"),
        ]:
            reference = catalog["references"][reference_id]
            metadata = {
                "reference_id": reference_id,
                "residue_order": catalog["canonical_order"],
                "unit": reference["unit"],
            }
            values = {a: reference["values"][a] for a in catalog["canonical_order"]}
            examples.append(
                {
                    "symbol": symbol,
                    "source_code": f"PROTEIN_FEATURE_REFERENCES = { {symbol: metadata}!r}\n{symbol} = {values!r}\n",
                }
            )
        result["reference_constants"] = {
            "metadata_keys_exactly": ["reference_id", "residue_order", "unit"],
            "rules": [
                "Each declaration key must exactly match the separately initialized module-level variable name, e.g. HYDROPATHY is different from HYDRO.",
                "The metadata has exactly three keys. Do not put values or amino-acid entries inside the metadata; values belong in the separate variable.",
                "Initialize each declared variable once, directly from a literal dict/list/tuple or numpy array/asarray of literals. Aliases, metadata lookups and computed initializers are unsupported.",
                "Use a single PROTEIN_FEATURE_REFERENCES mapping. If using both examples, merge their entries into that mapping rather than assigning it twice.",
                "These examples declare reference constants only. Choose and implement your own model. Literal parity does not establish predictive value or biological interpretation.",
            ],
            "standalone_examples": examples,
        }
    return result


def reference_repair_hints(source):
    """Describe literal declaration mistakes; never execute candidate code."""
    hints = [
        "Each reference metadata entry must have EXACTLY reference_id, residue_order, unit; no values or amino-acid keys. Put the numerical table in a separate module-level variable with the SAME name as the declaration key.",
        "Initialize that variable once with a literal dict/list/tuple or numpy array/asarray of literals, not an alias, metadata lookup or computation. Use the reference_constants examples from inspect_model_code_task; merge metadata entries into one mapping.",
    ]
    try:
        tree = ast.parse(source)
        assignments = {}
        for node in tree.body:
            targets = (
                node.targets
                if isinstance(node, ast.Assign)
                else [node.target]
                if isinstance(node, ast.AnnAssign)
                else []
            )
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, []).append(node.value)
        declarations = assignments.get("PROTEIN_FEATURE_REFERENCES", [])
        if len(declarations) != 1:
            return hints + [f"Found {len(declarations)} declarations; use exactly one literal mapping."]
        declared = ast.literal_eval(declarations[0])
        if not isinstance(declared, dict):
            return hints
        for symbol, metadata in list(declared.items())[:6]:
            if not isinstance(symbol, str) or not isinstance(metadata, dict):
                continue
            required = {"reference_id", "residue_order", "unit"}
            extra = sorted(set(metadata) - required, key=repr)
            missing = sorted(required - set(metadata))
            if extra or missing:
                hints.append(f"{symbol}: unexpected metadata keys={extra}; missing keys={missing}.")
            initializers = assignments.get(symbol, [])
            if len(initializers) != 1:
                hints.append(
                    f"{symbol}: found {len(initializers)} module-level initializations; the declaration must match exactly one variable by name."
                )
            elif isinstance(initializers[0], (ast.Name, ast.Subscript, ast.BinOp, ast.DictComp, ast.ListComp)):
                hints.append(
                    f"{symbol}: initializer is {type(initializers[0]).__name__}; replace it with the separate literal table from the catalog."
                )
    except (ValueError, TypeError, SyntaxError, RecursionError):
        hints.append("Use a single literal declaration mapping and check its Python syntax.")
    return hints


def code_failure_message(exc, stage, source, *, repeated=False):
    """Actionable feedback within the existing four-field, 3000-character contract."""
    original = str(exc)
    hints = []
    if repeated:
        hints.append(
            "IDENTICAL SOURCE: this source was already attempted in this episode. Changing rationale alone does not change code, and this retry still consumes an attempt."
        )
    if stage == "source_validation" and "Protein reference declarations failed:" in original:
        hints.extend(reference_repair_hints(source))
    elif isinstance(exc, SyntaxError):
        hints.append(
            "Correct the reported Python syntax/indentation in the complete source; do not submit Markdown fences or incomplete code fragments."
        )
    elif "NameError" in original or "not defined" in original:
        hints.append(
            "Import every allowed module explicitly and define every helper used by fit/predict. For example, np.mean needs import numpy as np, math.sqrt needs import math, and json needs import json."
        )
    elif stage == "prediction_validation":
        hints.append(
            "Check predict against the inspected ABI: finite numeric outputs, exactly one output per input in the same order; classifiers need one probability vector per input in class_labels order, summing to one."
        )
    elif stage == "execution":
        hints.append(
            "Repair the reported runtime error in the complete source. Export only JSON-compatible parameters from fit, then reconstruct predictions from those parameters within the existing CPU/memory limits."
        )
    if not hints:
        return original[:3000]
    return (original[:1400] + "\nRepair guidance: " + " ".join(hints))[:3000]


def info():
    return json.loads((BUNDLE / "task.json").read_text())


def model_api():
    plugin = info().get("plugin_id", "generated_python.v1")
    if plugin == "generated_python.v1":
        return generated_model
    if plugin == "generated_sequence.v1":
        import generated_sequence

        return generated_sequence
    if plugin in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        from context_interfaces import module_for

        return module_for(plugin)
    if plugin == "generated_features.v1":
        import generated_features

        return generated_features
    if plugin == "generated_classifier.large.v1":
        import generated_classifier_large

        return generated_classifier_large
    if plugin == "generated_classifier.v1":
        import generated_classifier

        return generated_classifier
    raise ValueError("Unknown generated model plugin")


def read_input(name):
    path = ROOT / name
    if path.is_symlink():
        raise ValueError("symlink input rejected")
    rows = json.loads(path.read_text())
    if digest(rows) != info()["inputs"][name]:
        raise ValueError("frozen task input changed")
    return rows


def save(name, value):
    path = ROOT / name
    if path.exists() or path.is_symlink():
        raise ValueError("artifact already exists")
    temporary = ROOT / (name + ".writing")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Atomic, exclusive publication: never replace an existing evidence file.
    os.link(temporary, path)
    temporary.unlink()
    sync_directory()


def sync_directory():
    descriptor = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ordinary_json(name):
    path = ROOT / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked development evidence")
    value = json.loads(path.read_text())
    digest(value)
    return value


def file_sha(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Missing or symlinked development evidence")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ledger_binding():
    return {
        "task_sha256": digest(info()),
        "implementation_sha256": digest({p.name: file_sha(p) for p in sorted(BUNDLE.glob("*.py"))}),
    }


def store_ledger(state):
    path, temporary = ROOT / LEDGER, ROOT / (LEDGER + ".writing")
    if path.is_symlink():
        raise ValueError("Symlinked development ledger")
    history = "code_ledger_" + digest(state) + ".json"
    if (ROOT / history).exists():
        if ordinary_json(history) != state:
            raise ValueError("Development history changed")
    else:
        save(history, state)
    with temporary.open("x") as stream:
        stream.write(json.dumps(state, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory()


def load_ledger():
    if (ROOT / (LEDGER + ".writing")).exists():
        raise ValueError("Incomplete development ledger write; host reconciliation required")
    path = ROOT / LEDGER
    if not path.exists() and not path.is_symlink():
        if list(ROOT.glob("code_attempt_*.json")) or any(
            (ROOT / n).exists() for n in ("candidate.json", "source_provenance.json")
        ):
            raise ValueError("Existing unbound development evidence requires a new allocation or host reconciliation")
        state = {"schema": DEVELOPMENT_LEDGER_SCHEMA, "binding": ledger_binding(), "attempts": [], "submission": None}
        store_ledger(state)
    state = ordinary_json(LEDGER)
    if (
        set(state) != {"schema", "binding", "attempts", "submission"}
        or state["schema"] != DEVELOPMENT_LEDGER_SCHEMA
        or state["binding"] != ledger_binding()
    ):
        raise ValueError("Development ledger task or implementation binding changed")
    if ordinary_json("code_ledger_" + digest(state) + ".json") != state:
        raise ValueError("Development ledger history differs")
    attempts = state["attempts"]
    if not isinstance(attempts, list) or len(attempts) > 2:
        raise ValueError("Invalid durable development budget")
    tested = {}
    expected_artifacts = set()
    for number, item in enumerate(attempts, 1):
        if (
            type(item.get("number")) is not int
            or item["number"] != number
            or item.get("status") not in {"reserved", "completed"}
        ):
            raise ValueError("Invalid development attempt order or status")
        if hashlib.sha256(item["source_code"].encode()).hexdigest() != item["source_sha256"]:
            raise ValueError("Reserved source fingerprint differs")
        if "source_origin" in item:
            from library_source import validate_source_origin

            validate_source_origin(BUNDLE, info(), item["source_code"], item["source_origin"])
        if item["status"] == "reserved":
            if number != len(attempts):
                raise ValueError("Execution continued after an unresolved attempt")
            continue
        name = f"code_attempt_{number}.json"
        expected_artifacts.add(name)
        record = ordinary_json(name)
        if (
            file_sha(ROOT / name) != item.get("artifact_sha256")
            or record.get("source_code") != item["source_code"]
            or record.get("rationale") != item["rationale"]
            or record.get("source_origin") != item.get("source_origin")
        ):
            raise ValueError("Completed development evidence changed")
        if record.get("status", "tested_not_published") == "tested_not_published":
            tested[item["source_sha256"]] = {
                "recipe": {"source_code": item["source_code"], "seed": 314159},
                "rationale": item["rationale"],
                "training_feedback": record["training_feedback"],
                **({"source_origin": item["source_origin"]} if "source_origin" in item else {}),
            }
        elif record.get("status") != "development_failed":
            raise ValueError("Unknown development feedback status")
    actual = {p.name for p in ROOT.glob("code_attempt_*.json")}
    pending_name = f"code_attempt_{len(attempts)}.json" if attempts and attempts[-1]["status"] == "reserved" else None
    if actual - expected_artifacts - {pending_name} or expected_artifacts - actual:
        raise ValueError("Development attempt artifact coverage differs")
    submission = state["submission"]
    if submission is not None:
        if submission.get("status") not in {"reserved", "completed"} or submission.get("source_sha256") not in tested:
            raise ValueError("Submission has no successful bound source")
        if submission["status"] == "completed":
            hashes = submission.get("artifact_sha256", {})
            if set(hashes) != {"candidate.json", "source_provenance.json"} or any(
                file_sha(ROOT / name) != expected for name, expected in hashes.items()
            ):
                raise ValueError("Completed submission evidence changed")
    elif any((ROOT / name).exists() for name in ("candidate.json", "source_provenance.json")):
        raise ValueError("Unreserved candidate evidence")
    return state, tested


@contextlib.contextmanager
def development_lock():
    descriptor = os.open(ROOT / ".model_development.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield load_ledger()
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def settled(state):
    if (
        any(row["status"] == "reserved" for row in state["attempts"])
        or (state["submission"] or {}).get("status") == "reserved"
    ):
        raise ValueError("Unresolved development execution; host reconciliation required before further work")


def development_state(state, tested):
    return {
        "schema": DEVELOPMENT_LEDGER_SCHEMA,
        "ledger_sha256": digest(state),
        "attempts_used": len(state["attempts"]),
        "attempts_remaining": 2 - len(state["attempts"]),
        "submittable_sources": sorted(tested),
        "pending_attempts": [row["number"] for row in state["attempts"] if row["status"] == "reserved"],
        "submission_status": (state["submission"] or {}).get("status", "not_submitted"),
    }


@mcp.tool()
def inspect_model_code_task() -> dict:
    """Inspect measured training data and the Python fit/predict model ABI. No assessment labels."""
    task = info()
    result = {
        "phase": task["phase"],
        "scope": task["scope"],
        "abi": {
            "fit": "fit(sequences: list[str], targets: list[float], seed: int) -> JSON-serializable model state",
            "predict": "predict(model, sequences: list[str]) -> list of finite predictions, input order preserved",
        },
        "libraries": ["numpy", "sklearn", "math", "itertools", "collections", "statistics", "json"],
        "constraints": [
            "no filesystem, networking, dynamic code execution or private introspection",
            "deterministic fit under supplied seed",
            "no pickle: export numeric parameters as JSON",
            "infer length/features from training inputs rather than hardcoding this protein",
        ],
        "development_budget": {"training_feedback_attempts": 2, "submitted_candidates": 1},
    }
    if "development_handoff" in task:
        from development_handoff import FIELD, description

        handoff = description(task)
        if handoff is not None:
            result[FIELD] = handoff
    if task.get("plugin_id") in {"generated_classifier.v1", "generated_classifier.large.v1"}:
        result["abi"] = {
            "fit": "fit(sequences: list[str], targets: list[int], seed: int) -> JSON-serializable model state",
            "predict": "predict(model, sequences: list[str]) -> list[list[float]], one probability vector per sequence, columns in scope.class_labels order, each row sums to one",
        }
        result["acceptance_metric"] = "brier_skill_vs_training_prior"
        result["constraints"].append(
            "CPU worker limits: 90 CPU seconds, 120 wall seconds and 4 GiB per operation; handle all input rows efficiently"
        )
    if task.get("plugin_id") == "generated_sequence.v1":
        result["sequence_layout"] = {
            "kind": "variable_length",
            "minimum": task["scope"]["minimum_sequence_length"],
            "maximum": task["scope"]["maximum_sequence_length"],
        }
        result["acceptance_metric"] = "mse_skill_vs_training_mean"
        result["constraints"].append(
            "Input proteins can have different lengths. Preserve full sequences; choose a variable-length representation without assuming an alignment."
        )
    if task.get("plugin_id") == "generated_features.v1":
        result["abi"] = {
            "fit": "fit(features: list[list[float]], targets: list[float], seed: int) -> JSON-serializable model state",
            "predict": "predict(model, features: list[list[float]]) -> one finite numeric prediction per row, same input order",
        }
        result["representation"] = task["scope"]["representation"]
        result["acceptance_metric"] = "mse_skill_vs_training_mean"
        result["constraints"].extend(
            [
                "Features are already computed by a frozen protein encoder; fit receives numeric vectors, not sequence strings.",
                "Infer feature dimension; choose preprocessing and a numerical estimator yourself; preprocessing must use training data only.",
                "CPU worker limits: 90 CPU seconds, 120 wall seconds and 4 GiB per operation.",
            ]
        )
    if task.get("plugin_id") in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        result["abi"] = {
            "fit": "fit(features: list[list[float]], targets: list[float], seed: int) -> JSON-serializable model state",
            "predict": "predict(model, features: list[list[float]]) -> one finite numeric prediction per row, same input order",
        }
        result["covariates"] = task["scope"]["covariates"]
        result["feedback_split"] = "whole independent groups within training; no group crosses the development split"
        result["acceptance_metric"] = "mse_skill_vs_training_mean"
        result["constraints"].extend(
            [
                "Numeric covariates describe measured conditions for the scoped protein. Column order and units are fixed in scope.covariates.",
                "Observation, protein and group identifiers are host metadata, never predictor features. The worker receives only numeric values and permitted training targets.",
                "All preprocessing is fitted on training data only. Training feedback holds out whole source-defined independent groups.",
                "CPU worker limits: 90 CPU seconds, 120 wall seconds and 4 GiB per operation.",
            ]
        )
    if task.get("plugin_id") == "generated_antibody.v1":
        result["representation"] = task["scope"]["representation"]
        result["constraints"].append(
            "Paired-antibody measurements preserve separate heavy/light chains, target construct digest and source patents. The worker receives the frozen numeric descriptor map; these inputs do not include final labels. Reported molar concentrations are not relabeled KD, and absent conditions remain unreported."
        )
    if task.get("plugin_id") == "generated_assay.v1":
        result["representation"] = task["scope"]["representation"]
        result["constraints"].append(
            "Assay observations can repeat a protein under different substrates or conditions; proteins may have different sequence lengths. The worker receives the source-bound numeric representation. Protein, sequence, substrate, publication and group identities stay on the host; never infer labels from them."
        )
    if task.get("training_baseline_feedback", "none") != "none":
        from context_feedback_dispatch import description, policy_for

        policy_for(task)
        if task["phase"] != "develop":
            raise ValueError("Baseline feedback is unavailable during reuse")
        result["training_baseline_feedback"] = description(task["scope"], policy=policy_for(task))
    if task["phase"] == "develop" and task.get("plugin_id") not in {
        "generated_features.v1",
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        result["residue_reference_catalog"] = reference_catalog()
    if task["phase"] == "develop":
        result["authoring_contract"] = authoring_contract(
            include_references=task.get("plugin_id")
            not in {
                "generated_features.v1",
                "generated_context.v1",
                "generated_context.v2",
                "generated_context.v3",
                "generated_assay.v1",
                "generated_antibody.v1",
            }
        )
        rows = read_input("training.json")
        model_api().validate_rows(rows, task["scope"], labels=True)
        result["training"] = {
            "rows": len(rows),
            "examples": rows[:3],
            "target_min": min(r["target"] for r in rows),
            "target_max": max(r["target"] for r in rows),
        }
        if task.get("plugin_id") == "generated_features.v1":
            result["training"]["examples"] = model_api().training_examples(rows)
        if task.get("plugin_id") in {"generated_classifier.v1", "generated_classifier.large.v1"}:
            result["training_class_counts"] = [
                sum(r["target"] == k for r in rows) for k in range(len(task["scope"]["class_labels"]))
            ]
        with development_lock() as (state, tested):
            result["development_state"] = development_state(state, tested)

    return result


@mcp.tool()
def read_model_experience() -> dict:
    """Read the permitted prior-model experience. It may be empty, summaries only, or summaries plus source. No target acceptance labels or fitted donor weights are provided."""
    task = info()
    if task["phase"] != "develop":
        raise ValueError("experience is available only during development")
    if "experience_sha256" not in task:
        return {"schema": 1, "condition": "none", "entries": [], "limitation": LIMITATION}
    packet = json.loads((BUNDLE / "experience.json").read_text())
    if digest(packet) != task["experience_sha256"]:
        raise ValueError("frozen model experience changed")
    return validate_projection(packet)


@mcp.tool()
def read_model_library_experience() -> dict:
    """Read frozen cross-task implementation experience: empty, summary or source. Receiving tasks require a new fit and independent acceptance."""
    task = info()
    if task["phase"] != "develop" or "library_experience_sha256" not in task:
        raise ValueError("No library experience admitted for this episode")
    packet = json.loads((BUNDLE / "library_experience.json").read_text())
    if digest(packet) != task["library_experience_sha256"]:
        raise ValueError("Frozen library experience changed")
    return validate_projection(packet)


@mcp.tool()
def evaluate_model_code(source_code: str, rationale: str) -> dict:
    """Test generated source with training-only feedback. Every source attempt, including failed code, consumes one of two attempts. Failed attempts are never submittable."""
    return evaluate_source(source_code, rationale)


@mcp.tool()
def evaluate_library_model_code(experience_id: str, rationale: str) -> dict:
    """Refit an unchanged source from read_model_library_experience by experience_id. Uses only current training data, no donor weights. Shares the two-attempt budget with evaluate_model_code. Submit a successful source hash separately; independent acceptance follows."""
    from library_source import resolve_library_source

    source, origin = resolve_library_source(BUNDLE, info(), experience_id)
    return evaluate_source(source, rationale, source_origin=origin)


def evaluate_source(source_code, rationale, *, source_origin=None):
    task = info()
    if task["phase"] != "develop":
        raise ValueError("code-development budget exhausted")
    # Corrupt input/scope and unsupported plugins must remain fatal, outside the
    # candidate-repair boundary.
    rows = read_input("training.json")
    api = model_api()
    api.validate_rows(rows, task["scope"], labels=True)
    if task.get("plugin_id") in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        train, dev = api.development_split(rows)
    else:
        ordered = sorted(rows, key=lambda r: digest({"seed": 314159, "sequence": r["sequence"]}))
        n = max(10, int(len(rows) * 0.8))
        train, dev = ordered[:n], ordered[n:]
    source_sha256 = hashlib.sha256(source_code.encode()).hexdigest()
    with development_lock() as (state, tested):
        settled(state)
        if len(state["attempts"]) >= 2:
            raise ValueError("code-development budget exhausted")
        if state["submission"] is not None:
            raise ValueError("one submitted candidate permitted")
        number = len(state["attempts"]) + 1
        attempt = {
            "number": number,
            "source_code": source_code,
            "source_sha256": source_sha256,
            "rationale": rationale,
            "status": "reserved",
        }
        if source_origin is not None:
            attempt["source_origin"] = source_origin
        state["attempts"].append(attempt)
        store_ledger(state)

        def finish(feedback, failed=False):
            record = {"source_code": source_code, "rationale": rationale, "training_feedback": feedback}
            result = {
                "source_sha256": source_sha256,
                "training_only_feedback": feedback,
                "attempts_remaining": 2 - number,
                "status": "development_failed" if failed else "tested_not_published",
            }
            if source_origin is not None:
                record["source_origin"] = source_origin
                result["source_origin"] = source_origin
            if failed:
                record.update(status="development_failed", submittable_sources=sorted(tested))
                result["submittable_sources"] = sorted(tested)
            name = f"code_attempt_{number}.json"
            save(name, record)
            attempt.update(status="completed", artifact_sha256=file_sha(ROOT / name))
            store_ledger(state)
            return result

        def failure(exc, stage):
            return finish(
                {
                    "status": "failed",
                    "stage": stage,
                    "error_type": type(exc).__name__,
                    "message": code_failure_message(
                        exc,
                        stage,
                        source_code,
                        repeated=any(previous["source_sha256"] == source_sha256 for previous in state["attempts"][:-1]),
                    ),
                },
                failed=True,
            )

        try:
            guard_source(source_code)
        except (ValueError, SyntaxError) as exc:
            return failure(exc, "source_validation")
        recipe = {"source_code": source_code, "seed": 314159}
        try:
            model = api.fit_candidate(train, task["scope"], recipe)
            metrics = api.evaluate(model, dev, task["scope"])
        except CandidateCodeError as exc:
            return failure(exc, "execution")
        except ModelOutputError as exc:
            return failure(exc, "prediction_validation")
        if task.get("training_baseline_feedback", "none") != "none":
            from context_feedback_dispatch import comparison, policy_for

            policy_for(task)
            metrics["baseline_comparison"] = comparison(rows, task["scope"], metrics, policy=policy_for(task))
        return finish(metrics)


@mcp.tool()
def submit_model_code(source_sha256: str, rationale: str) -> dict:
    """Choose one successfully tested source version and fit it to all permitted training rows. Writes candidate.json; independent host validation follows later."""
    if info()["phase"] != "develop":
        raise ValueError("source must first pass a training-only code test")
    handoff = {}
    if "development_handoff" in info():
        from development_handoff import receipt_fields

        handoff = receipt_fields(info(), source_sha256)
    with development_lock() as (state, tested):
        settled(state)
        if source_sha256 not in tested:
            raise ValueError("source must first pass a training-only code test")
        previous = state["submission"]
        if previous is not None:
            if previous["source_sha256"] != source_sha256 or previous["rationale"] != rationale:
                raise ValueError("one submitted candidate permitted; submission arguments differ")
            return previous["receipt"]
        chosen = tested[source_sha256]
        state["submission"] = {"source_sha256": source_sha256, "rationale": rationale, "status": "reserved"}
        store_ledger(state)
        model = model_api().fit_candidate(read_input("training.json"), info()["scope"], chosen["recipe"])
        save("candidate.json", model)
        save(
            "source_provenance.json",
            {
                "source_sha256": source_sha256,
                "rationale": rationale,
                "training_feedback": chosen["training_feedback"],
                "training_feedback_attempts": len(state["attempts"]),
                "origin": "admitted_library_source_refit"
                if "source_origin" in chosen
                else "agent_tool_call_source_code",
                **({"source_origin": chosen["source_origin"]} if "source_origin" in chosen else {}),
                "source_reference_audit": audit_source_references(chosen["recipe"]["source_code"]),
            },
        )
        receipt = {
            "artifact": "candidate.json",
            "source_sha256": source_sha256,
            "training_rows": model["training_rows"],
            "status": "candidate",
            "registered": False,
            **handoff,
        }
        state["submission"].update(
            status="completed",
            receipt=receipt,
            artifact_sha256={name: file_sha(ROOT / name) for name in ("candidate.json", "source_provenance.json")},
        )
        store_ledger(state)
        return receipt


def compatible_packet():
    task = info()
    path = BUNDLE / "registered/capability.json"
    if task["phase"] != "reuse" or not path.exists():
        return None
    packet = json.loads(path.read_text())
    if (
        digest(packet) != task["capability_sha256"]
        or digest(packet["model"]) != packet["model_sha256"]
        or packet["model"]["scope"] != task["scope"]
        or packet["model"].get("plugin_id") != task.get("plugin_id", "generated_python.v1")
    ):
        raise ValueError("capability integrity or applicability mismatch")
    return packet


@mcp.tool()
def discover_generated_models() -> dict:
    """Discover validated generated-code models compatible with this protein task."""
    packet = compatible_packet()
    return {
        "models": []
        if packet is None
        else [
            {
                "model_id": packet["model_id"],
                "scope": packet["model"]["scope"],
                "source_sha256": packet["model"]["source_sha256"],
                "status": "registered",
                "assessment": packet["assessment"]["metrics"],
                **(
                    {"source_reference_audit": packet["assessment"]["source_reference_audit"]}
                    if "source_reference_audit" in packet["assessment"]
                    else {}
                ),
            }
        ]
    }


@mcp.tool()
def predict_with_generated_model(model_id: str) -> dict:
    """Run a discovered validated code model in a networkless worker and save every prediction."""
    packet = compatible_packet()
    if packet is None or model_id != packet["model_id"]:
        raise ValueError("validated model unavailable")
    predictions = model_api().predict(packet["model"], read_input("inputs.json"), info()["scope"])
    save("predictions.json", predictions)
    return {
        "artifact": "predictions.json",
        "model_id": model_id,
        "rows": len(predictions),
        "prediction_sha256": digest(predictions),
    }


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
