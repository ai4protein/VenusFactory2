"""Bounded public tools for developing and reusing paper computation steps.

This service never receives private acceptance references or host registry paths.
Its artifacts are submissions; registration remains a separate host decision.
"""

import hashlib
import json
from pathlib import Path

from .code_sandbox import CandidateCodeError
from .paper_step import compare, digest, execute_step, validated_source
from .storage import atomic_json, file_lock, safe_child, sha256_file

TOOLS = frozenset(
    {
        "inspect_paper_step_task",
        "evaluate_paper_step",
        "submit_paper_step",
        "discover_paper_steps",
        "execute_registered_paper_step",
    }
)


def source_hash(source):
    return hashlib.sha256(source.encode()).hexdigest()


def verify_execution(source, inputs, record):
    expected = {
        "operation": "predict",
        "source_sha256": source_hash(validated_source(source)),
        "request_sha256": digest(
            {"operation": "predict", "input_kind": "features", "features": inputs, "model": None, "seed": 0}
        ),
        "result_sha256": digest({"predictions": record["output"]}),
        "worker_sha256": sha256_file(Path(__file__).with_name("code_worker.py")),
        "runtime_probe_sha256": sha256_file(Path(__file__).with_name("numeric_runtime.py")),
    }
    evidence = record["execution"]
    if any(evidence.get(k) != v for k, v in expected.items()) or not evidence.get("runtime", {}).get(
        "packages", {}
    ).get("numpy"):
        raise ValueError("Paper step numerical witness does not match source, inputs and output")


class PaperStepTools:
    def __init__(self, workspace, capability):
        self.workspace = Path(workspace).resolve()
        self.capability = Path(capability).resolve()

    def read(self, root, name):
        return json.loads(safe_child(root, name).read_text())

    def save(self, name, value):
        path = safe_child(self.workspace, name)
        if path.exists():
            raise ValueError("Paper step artifact already exists")
        atomic_json(path, value)

    def task(self):
        task = self.read(self.capability, "step_task.json")
        if (
            task.get("schema") != 1
            or task.get("phase") not in {"develop", "reuse"}
            or task.get("development_attempts") != 2
        ):
            raise ValueError("Invalid frozen paper step tool task")
        return task

    def inspect(self):
        task = self.task()
        return {
            "phase": task["phase"],
            "task": task["public_task"],
            "development_attempts": 2,
            "registration": "Independent host acceptance after this session; these tools cannot register capabilities",
        }

    def attempt(self, number):
        if type(number) is not int or number not in {1, 2}:
            raise ValueError("Invalid development attempt")
        task = self.task()
        if task["phase"] != "develop":
            raise ValueError("Development unavailable")
        record = self.read(self.workspace, f"step_attempt_{number}.json")
        reservation = self.read(self.workspace, f"step_reservation_{number}.json")
        if record["source_sha256"] != source_hash(record["source_code"]) or reservation != {
            "source_sha256": record["source_sha256"]
        }:
            raise ValueError("Development reservation or source differs")
        if record["status"] == "tested_not_registered":
            cases = task["public_task"]["development_cases"]
            if len(record["results"]) != len(cases):
                raise ValueError("Incomplete development results")
            for case, result in zip(cases, record["results"], strict=True):
                if result["case_id"] != case["id"]:
                    raise ValueError("Development case identity differs")
                verify_execution(record["source_code"], case["inputs"], result)
                checked = compare(result["output"], case["expected"], **task["public_task"]["tolerance"])
                if checked != result["comparison"]:
                    raise ValueError("Development comparison differs")
        elif record["status"] != "development_failed" or not record.get("error_type"):
            raise ValueError("Invalid failed development record")
        return record

    def evaluation_receipt(self, number):
        record = self.attempt(number)
        passed = record["status"] == "tested_not_registered" and all(
            x["comparison"]["passed"] for x in record["results"]
        )
        return {
            "attempt": number,
            "source_sha256": record["source_sha256"],
            "status": record["status"],
            "development_passed": passed,
            "attempts_remaining": 2 - number,
            "artifact": f"step_attempt_{number}.json",
            "feedback": [x["comparison"] for x in record.get("results", [])],
            "error": {"type": record["error_type"], "message": record["message"]}
            if record["status"] == "development_failed"
            else None,
        }

    def evaluate(self, source_code, rationale):
        task = self.task()
        if task["phase"] != "develop":
            raise ValueError("Development unavailable")
        if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 4000:
            raise ValueError("A bounded rationale is required")
        with file_lock(safe_child(self.workspace, ".paper_step_tools.lock")):
            numbers = [n for n in (1, 2) if safe_child(self.workspace, f"step_reservation_{n}.json").exists()]
            if len(numbers) >= 2:
                raise ValueError("Paper step development budget exhausted")
            number = len(numbers) + 1
            sha = source_hash(source_code)
            self.save(f"step_reservation_{number}.json", {"source_sha256": sha})
            record = {"source_code": source_code, "source_sha256": sha, "rationale": rationale, "results": []}
            try:
                validated_source(source_code)
                for case in task["public_task"]["development_cases"]:
                    execution = execute_step(
                        source_code, case["inputs"], timeout_s=task["public_task"]["budget"]["operation_seconds"]
                    )
                    record["results"].append(
                        {
                            "case_id": case["id"],
                            **execution,
                            "comparison": compare(
                                execution["output"], case["expected"], **task["public_task"]["tolerance"]
                            ),
                        }
                    )
                record["status"] = "tested_not_registered"
            except (ValueError, SyntaxError, CandidateCodeError) as exc:
                record.update(status="development_failed", error_type=type(exc).__name__, message=str(exc)[:2000])
            self.save(f"step_attempt_{number}.json", record)
            return self.evaluation_receipt(number)

    def submission_receipt(self):
        candidate = self.read(self.workspace, "step_candidate.json")
        attempt = self.attempt(candidate["development_attempt"])
        if not self.evaluation_receipt(candidate["development_attempt"])["development_passed"]:
            raise ValueError("Submitted source did not pass development")
        expected = {
            "source_code": attempt["source_code"],
            "source_sha256": attempt["source_sha256"],
            "rationale": attempt["rationale"],
            "development_attempt": candidate["development_attempt"],
            "registration": "candidate_only",
        }
        if candidate != expected:
            raise ValueError("Submitted source differs from successful development")
        return {
            "artifact": "step_candidate.json",
            "source_sha256": candidate["source_sha256"],
            "status": "candidate",
            "registered": False,
        }

    def submit(self, source_sha256):
        if self.task()["phase"] != "develop":
            raise ValueError("Submission unavailable")
        with file_lock(safe_child(self.workspace, ".paper_step_tools.lock")):
            for n in (1, 2):
                if not safe_child(self.workspace, f"step_attempt_{n}.json").exists():
                    continue
                attempt = self.attempt(n)
                if attempt["source_sha256"] == source_sha256 and self.evaluation_receipt(n)["development_passed"]:
                    self.save(
                        "step_candidate.json",
                        {
                            "source_code": attempt["source_code"],
                            "source_sha256": source_sha256,
                            "rationale": attempt["rationale"],
                            "development_attempt": n,
                            "registration": "candidate_only",
                        },
                    )
                    return self.submission_receipt()
            raise ValueError("Source must pass a development test before submission")

    def packet(self):
        task = self.task()
        if task["phase"] != "reuse":
            raise ValueError("Registered step unavailable")
        packet = self.read(self.capability, "step_capability.json")
        if (
            digest(packet) != task["capability_sha256"]
            or packet["interface"] != task["public_task"]["interface"]
            or packet["scope"] != task["public_task"]["scope"]
        ):
            raise ValueError("Registered step integrity or applicability differs")
        if packet["verification"].get("passed") is not True:
            raise ValueError("Registered step lacks host acceptance")
        return packet

    def discover(self):
        capabilities = []
        if self.task()["phase"] == "reuse":
            packet = self.packet()
            capabilities.append(
                {
                    "capability_id": packet["capability_id"],
                    "source_sha256": source_hash(packet["source_code"]),
                    "scope": packet["scope"],
                    "interface": packet["interface"],
                }
            )
        return {"capabilities": capabilities}

    def reuse_inputs(self):
        task = self.task()
        value = self.read(self.workspace, "step_inputs.json")
        if digest(value) != task["inputs_sha256"]:
            raise ValueError("Fresh step inputs changed")
        return value

    def reuse_receipt(self):
        task = self.task()
        packet = self.packet()
        inputs = self.reuse_inputs()
        record = self.read(self.workspace, "step_reuse.json")
        reservation = self.read(self.workspace, "step_reuse_reservation.json")
        if reservation != {"capability_id": packet["capability_id"], "case_id": task["case_id"]}:
            raise ValueError("Fresh execution reservation differs")
        if (
            record["case_id"] != task["case_id"]
            or record["capability_id"] != packet["capability_id"]
            or record["input_sha256"] != digest(inputs)
        ):
            raise ValueError("Fresh execution binding differs")
        verify_execution(packet["source_code"], inputs, record)
        return {
            "artifact": "step_reuse.json",
            "capability_id": packet["capability_id"],
            "case_id": task["case_id"],
            "input_sha256": digest(inputs),
            "output_sha256": digest(record["output"]),
            "status": "computed_not_independently_accepted",
        }

    def reuse(self, capability_id):
        task = self.task()
        packet = self.packet()
        if capability_id != packet["capability_id"]:
            raise ValueError("Unknown registered step")
        inputs = self.reuse_inputs()
        with file_lock(safe_child(self.workspace, ".paper_step_tools.lock")):
            self.save("step_reuse_reservation.json", {"capability_id": capability_id, "case_id": task["case_id"]})
            record = execute_step(
                packet["source_code"], inputs, timeout_s=task["public_task"]["budget"]["operation_seconds"]
            )
            self.save(
                "step_reuse.json",
                {"capability_id": capability_id, "case_id": task["case_id"], "input_sha256": digest(inputs), **record},
            )
            return self.reuse_receipt()
