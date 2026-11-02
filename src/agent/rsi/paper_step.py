"""Host-admitted, isolated numerical steps for extensible paper reproduction.

The host freezes examples, independent references and fresh reuse inputs before
code submission. Passing this contract proves only the declared computation.
LLM authorship and autonomous role execution require separate event evidence.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from pathlib import Path

from .code_sandbox import CandidateCodeError, capture_code_runs, guard_source, run_code
from .storage import atomic_json, file_lock, safe_child, sha256_file

WRAPPER = "\n\ndef fit(sequences, targets, seed):\n    return {}\n\ndef predict(model, sequences):\n    return reproduce(sequences)\n"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def implementation_hash():
    root = Path(__file__).parent
    return digest(
        {
            name: sha256_file(root / name)
            for name in [
                "paper_step.py",
                "code_sandbox.py",
                "code_worker.py",
                "numeric_runtime.py",
                "protein_references.py",
            ]
        }
    )


def validated_source(source):
    if not isinstance(source, str) or not 1 <= len(source.encode()) <= 58000:
        raise ValueError("Paper step source must contain 1..58000 bytes")
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reproduce"]
    if len(functions) != 1:
        raise ValueError("Required ABI: reproduce(inputs) -> JSON value")
    substantive = [
        n
        for n in functions[0].body
        if not isinstance(n, ast.Pass) and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))
    ]
    if not substantive:
        raise ValueError("Paper step must implement computation")
    args = functions[0].args
    if len(args.args) != 1 or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults:
        raise ValueError("Required ABI: reproduce(inputs) -> JSON value")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {"fit", "predict"}:
            raise ValueError("fit/predict are reserved for the trusted execution adapter")
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in {"fit", "predict"}:
            raise ValueError("fit/predict are reserved for the trusted execution adapter")
    wrapped = source + WRAPPER
    guard_source(wrapped)
    return wrapped


def compare(actual, expected, *, atol, rtol):
    """Compare complete JSON structure, with finite-number tolerances only."""
    digest(actual)
    digest(expected)
    errors = []
    count = 0
    numeric = 0
    max_difference = 0.0

    def error(path, reason):
        nonlocal count
        count += 1
        if len(errors) < 20:
            errors.append({"path": path, "reason": reason})

    def visit(a, b, path):
        nonlocal numeric, max_difference
        if type(b) in {int, float}:
            if type(a) not in {int, float}:
                error(path, "numeric type mismatch")
                return
            numeric += 1
            delta = abs(a - b)
            max_difference = max(max_difference, delta)
            if delta > atol + rtol * abs(b):
                error(path, "numeric tolerance exceeded")
        elif type(a) is not type(b):
            error(path, "JSON type mismatch")
        elif isinstance(b, dict):
            if set(a) != set(b):
                error(path, "object keys differ")
            for key in sorted(set(a) & set(b)):
                visit(a[key], b[key], path + "/" + key)
        elif isinstance(b, list):
            if len(a) != len(b):
                error(path, "array length differs")
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path + "/" + str(i))
        elif a != b:
            error(path, "exact value differs")

    visit(actual, expected, "$")
    return {
        "passed": count == 0,
        "mismatches": count,
        "details": errors,
        "numeric_values": numeric,
        "maximum_absolute_difference": max_difference,
    }


def validate_definition(value):
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "paper",
        "scope",
        "interface",
        "references",
        "cases",
        "budget",
        "tolerance",
    }:
        raise ValueError("Invalid paper step definition fields")
    if value["schema"] != 1 or type(value["schema"]) is not int:
        raise ValueError("Unsupported paper step schema")
    if set(value["paper"]) != {"title", "doi"} or any(
        not isinstance(x, str) or not x.strip() for x in value["paper"].values()
    ):
        raise ValueError("Paper title and DOI required")
    if set(value["scope"]) != {"covered_steps", "excluded_steps"}:
        raise ValueError("Explicit covered and excluded computational scope required")
    for items in value["scope"].values():
        if not isinstance(items, list) or not items or any(not isinstance(x, str) or not x.strip() for x in items):
            raise ValueError("Scope must contain nonempty step descriptions")
    if set(value["interface"]) != {"id", "description"} or any(
        not isinstance(x, str) or not x.strip() for x in value["interface"].values()
    ):
        raise ValueError("A named scientific input/output interface is required")
    if not isinstance(value["references"], dict) or not value["references"]:
        raise ValueError("Host-bound source data and independent reference implementation are required")
    for path, expected in value["references"].items():
        if not Path(path).is_absolute() or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("Reference paths must be absolute with SHA256 hashes")
    budget = value["budget"]
    if (
        set(budget) != {"candidates", "operation_seconds"}
        or type(budget["candidates"]) is not int
        or not 1 <= budget["candidates"] <= 20
    ):
        raise ValueError("Invalid candidate budget")
    if type(budget["operation_seconds"]) not in {int, float} or not 1 <= budget["operation_seconds"] <= 120:
        raise ValueError("Invalid bounded operation deadline")
    tolerance = value["tolerance"]
    if set(tolerance) != {"atol", "rtol"} or any(
        type(x) not in {int, float} or not math.isfinite(x) or not 0 <= x <= 0.01 for x in tolerance.values()
    ):
        raise ValueError("Invalid frozen numerical tolerance")
    if set(value["cases"]) != {"development", "acceptance", "reuse"}:
        raise ValueError("Development, independent acceptance and fresh reuse are required")
    ids, inputs = set(), set()
    for group in value["cases"].values():
        if not isinstance(group, list) or not 1 <= len(group) <= 8:
            raise ValueError("Each phase requires 1..8 admitted cases")
        for case in group:
            if (
                set(case) != {"id", "inputs", "expected"}
                or not isinstance(case["id"], str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", case["id"])
            ):
                raise ValueError("Invalid scientific case")
            if case["id"] in ids or digest(case["inputs"]) in inputs:
                raise ValueError("Case IDs and complete inputs must be distinct across phases")
            ids.add(case["id"])
            inputs.add(digest(case["inputs"]))
            if len(json.dumps(case, allow_nan=False).encode()) > 8 * 1024**2:
                raise ValueError("A paper step case exceeds the bounded JSON input size")
    digest(value)
    return value


def execute_step(source, inputs, *, timeout_s):
    wrapped = validated_source(source)
    request = {"operation": "predict", "input_kind": "features", "features": inputs, "model": None, "seed": 0}
    with capture_code_runs() as records:
        output = run_code(wrapped, request, timeout_s=timeout_s)["predictions"]
    digest(output)
    if len(records) != 1:
        raise ValueError("Missing exact execution evidence")
    return {"output": output, "execution": records[0]}


class PaperStepLifecycle:
    """Private host state. Never mount this directory in an agent workspace."""

    def __init__(self, root, definition):
        requested = Path(root).absolute()
        if any(p.is_symlink() for p in [requested, *requested.parents]):
            raise ValueError("Host state cannot traverse symlinks")
        self.root = requested.resolve()
        self.definition = json.loads(json.dumps(validate_definition(definition), allow_nan=False))
        self.binding = {"schema": 1, "definition": self.definition, "implementation_sha256": implementation_hash()}
        self.root.mkdir(parents=True, exist_ok=True)
        with file_lock(safe_child(self.root, ".step.lock")):
            self._save("admission.json", self.binding, same_ok=True)
        self._check()

    def _save(self, name, value, *, same_ok=False):
        path = safe_child(self.root, name)
        if path.exists():
            if same_ok and self._read(name) == value:
                return
            raise ValueError("Immutable paper step artifact already exists: " + name)
        atomic_json(path, {"payload": value, "sha256": digest(value)})
        path.chmod(0o444)

    def _read(self, name):
        record = json.loads(safe_child(self.root, name).read_text())
        if set(record) != {"payload", "sha256"} or digest(record["payload"]) != record["sha256"]:
            raise ValueError("Paper step evidence hash mismatch")
        return record["payload"]

    def _check(self):
        if (
            self._read("admission.json") != self.binding
            or implementation_hash() != self.binding["implementation_sha256"]
        ):
            raise ValueError("Paper step admission or implementation changed")
        for path, expected in self.definition["references"].items():
            if sha256_file(Path(path)) != expected:
                raise ValueError("Bound scientific reference changed")

    def public_task(self):
        self._check()
        return {key: self.definition[key] for key in ["paper", "scope", "interface", "budget", "tolerance"]} | {
            "abi": "reproduce(inputs) -> JSON value with the declared structure",
            "development_cases": self.definition["cases"]["development"],
            "acceptance_cases": len(self.definition["cases"]["acceptance"]),
            "reuse_cases": len(self.definition["cases"]["reuse"]),
            "claim_limit": "Defined numerical computation only; neither full-paper replication nor autonomous authorship is established.",
        }

    def propose(self, source, *, declared_origin):
        with file_lock(safe_child(self.root, ".step.lock")):
            self._check()
            if not isinstance(declared_origin, str) or not declared_origin.strip() or len(declared_origin) > 256:
                raise ValueError("Explicit declared code origin required")
            reservations = list(safe_child(self.root, "attempts").glob("*/reservation.json"))
            if len(reservations) >= self.definition["budget"]["candidates"]:
                raise ValueError("Candidate budget exhausted")
            number = len(reservations) + 1
            base = f"attempts/{number:04d}"
            payload = {
                "source_code": source,
                "declared_origin": declared_origin,
                "agent_authorship_verified": False,
                "admission_sha256": digest(self.binding),
                "attempt": number,
            }
            identity = digest(payload)
            self._save(base + "/reservation.json", {"candidate_id": identity})
            try:
                validated_source(source)
            except (ValueError, SyntaxError) as exc:
                self._save(
                    base + "/failure.json",
                    {"candidate": payload, "error_type": type(exc).__name__, "message": str(exc)[:2000]},
                )
                raise
            self._save("candidates/" + identity + ".json", payload)
            return identity

    def _candidate(self, identity):
        if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity):
            raise ValueError("Invalid candidate identity")
        candidate = self._read("candidates/" + identity + ".json")
        if digest(candidate) != identity or candidate["admission_sha256"] != digest(self.binding):
            raise ValueError("Candidate identity or admission mismatch")
        reservation = self._read(f"attempts/{candidate['attempt']:04d}/reservation.json")
        if reservation != {"candidate_id": identity}:
            raise ValueError("Candidate reservation mismatch")
        return candidate

    def _evaluate(self, source, case):
        record = execute_step(source, case["inputs"], timeout_s=self.definition["budget"]["operation_seconds"])
        return {
            "case_id": case["id"],
            "input_sha256": digest(case["inputs"]),
            **record,
            "comparison": compare(record["output"], case["expected"], **self.definition["tolerance"]),
        }

    def assess(self, identity):
        with file_lock(safe_child(self.root, ".step.lock")):
            self._check()
            candidate = self._candidate(identity)
            prefix = "assessments/" + identity
            self._save(prefix + "/reservation.json", {"candidate_id": identity})
            for case in self.definition["cases"]["acceptance"]:
                try:
                    evidence = self._evaluate(candidate["source_code"], case)
                except Exception as exc:
                    self._save(
                        prefix + "/failure.json",
                        {
                            "case_id": case["id"],
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:2000],
                            "candidate_failure": isinstance(exc, CandidateCodeError),
                        },
                    )
                    raise
                self._save(prefix + "/" + case["id"] + ".json", evidence)
            self._check()
            verdict = self._verify_assessment(identity)
            self._save(prefix + "/verdict.json", verdict)
            return verdict

    def _verify_execution(self, candidate, case, evidence):
        if evidence["case_id"] != case["id"] or evidence["input_sha256"] != digest(case["inputs"]):
            raise ValueError("Scientific case binding changed")
        execution = evidence["execution"]
        request = {
            "operation": "predict",
            "input_kind": "features",
            "features": case["inputs"],
            "model": None,
            "seed": 0,
        }
        expected = {
            "operation": "predict",
            "source_sha256": hashlib.sha256(validated_source(candidate["source_code"]).encode()).hexdigest(),
            "request_sha256": digest(request),
            "result_sha256": digest({"predictions": evidence["output"]}),
            "worker_sha256": sha256_file(Path(__file__).with_name("code_worker.py")),
            "runtime_probe_sha256": sha256_file(Path(__file__).with_name("numeric_runtime.py")),
        }
        if any(execution.get(k) != v for k, v in expected.items()) or not execution.get("runtime", {}).get(
            "packages", {}
        ).get("numpy"):
            raise ValueError("Numerical execution evidence differs from candidate, inputs or outputs")
        comparison = compare(evidence["output"], case["expected"], **self.definition["tolerance"])
        if comparison != evidence["comparison"]:
            raise ValueError("Recorded numerical comparison changed")
        return comparison["passed"]

    def _verify_assessment(self, identity):
        candidate = self._candidate(identity)
        if self._read("assessments/" + identity + "/reservation.json") != {"candidate_id": identity}:
            raise ValueError("Assessment reservation changed")
        results = []
        for case in self.definition["cases"]["acceptance"]:
            evidence = self._read("assessments/" + identity + "/" + case["id"] + ".json")
            results.append(
                {
                    "case_id": case["id"],
                    "passed": self._verify_execution(candidate, case, evidence),
                    "evidence_sha256": digest(evidence),
                }
            )
        return {
            "candidate_id": identity,
            "passed": all(r["passed"] for r in results),
            "cases": results,
            "agent_authorship_verified": False,
            "full_paper_reproduced": False,
        }

    def register(self, identity):
        with file_lock(safe_child(self.root, ".step.lock")):
            self._check()
            candidate = self._candidate(identity)
            verdict = self._verify_assessment(identity)
            if not verdict["passed"] or verdict != self._read("assessments/" + identity + "/verdict.json"):
                raise ValueError("Independent assessment did not pass")
            payload = {
                "schema": 1,
                "kind": "verified_paper_step",
                "candidate_id": identity,
                "source_sha256": hashlib.sha256(candidate["source_code"].encode()).hexdigest(),
                "admission_sha256": digest(self.binding),
                "assessment_sha256": digest(verdict),
            }
            cap_id = digest(payload)
            self._save("registered/" + cap_id + ".json", payload, same_ok=True)
            return cap_id

    def capability(self, identity):
        self._check()
        if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity):
            raise ValueError("Invalid registered capability identity")
        packet = self._read("registered/" + identity + ".json")
        if digest(packet) != identity or packet["admission_sha256"] != digest(self.binding):
            raise ValueError("Registered capability integrity changed")
        candidate = self._candidate(packet["candidate_id"])
        verdict = self._verify_assessment(packet["candidate_id"])
        if (
            not verdict["passed"]
            or digest(verdict) != packet["assessment_sha256"]
            or verdict != self._read("assessments/" + packet["candidate_id"] + "/verdict.json")
        ):
            raise ValueError("Registered capability assessment no longer valid")
        if hashlib.sha256(candidate["source_code"].encode()).hexdigest() != packet["source_sha256"]:
            raise ValueError("Registered source differs")
        return {
            "capability_id": identity,
            "source_code": candidate["source_code"],
            "paper": self.definition["paper"],
            "scope": self.definition["scope"],
            "interface": self.definition["interface"],
            "verification": verdict,
            "declared_origin": candidate["declared_origin"],
            "agent_authorship_verified": False,
        }

    def reuse(self, identity, case_id):
        with file_lock(safe_child(self.root, ".step.lock")):
            packet = self.capability(identity)
            cases = [r for r in self.definition["cases"]["reuse"] if r["id"] == case_id]
            if len(cases) != 1:
                raise ValueError("Fresh reuse requires an admitted case")
            prefix = "reuse/" + identity + "/" + case_id
            self._save(prefix + "/reservation.json", {"capability_id": identity, "case_id": case_id})
            try:
                evidence = self._evaluate(packet["source_code"], cases[0])
            except Exception as exc:
                self._save(prefix + "/failure.json", {"error_type": type(exc).__name__, "message": str(exc)[:2000]})
                raise
            self._check()
            self._save(prefix + "/evidence.json", evidence)
            return self.verify_reuse(identity, case_id)

    def verify_reuse(self, identity, case_id):
        packet = self.capability(identity)
        cases = [r for r in self.definition["cases"]["reuse"] if r["id"] == case_id]
        if len(cases) != 1:
            raise ValueError("Unknown admitted reuse case")
        prefix = "reuse/" + identity + "/" + case_id
        if self._read(prefix + "/reservation.json") != {"capability_id": identity, "case_id": case_id}:
            raise ValueError("Reuse reservation changed")
        evidence = self._read(prefix + "/evidence.json")
        candidate = self._candidate(packet["verification"]["candidate_id"])
        passed = self._verify_execution(candidate, cases[0], evidence)
        return {
            "capability_id": identity,
            "case_id": case_id,
            "passed": passed,
            "evidence_sha256": digest(evidence),
            "fresh_execution": True,
            "agent_authorship_verified": False,
            "full_paper_reproduced": False,
        }
