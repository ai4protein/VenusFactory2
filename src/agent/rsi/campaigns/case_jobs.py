"""Durable, bounded controller for agent-requested reproduction plugins.

The agent exchanges typed requests in its workspace. Host commands, model
weights, reports and job state remain outside that workspace. Only fixed
registered plugin entry points can run, without a shell or inherited keys.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

from agent.rsi.engine.case_evidence import canonical_claims
from .case_plugins import antibody_allocation, invocation, list_case_plugins, resource_for
from agent.rsi.engine.protein_model import digest
from agent.rsi.engine.storage import atomic_json, sha256_file


class CaseJobBroker:
    def __init__(
        self,
        *,
        root,
        workspace,
        output,
        allowed_plugins,
        max_jobs=1,
        timeout_s=1800,
        expected_parameters=None,
        maximum_backbones=8,
        maximum_sequences_per_backbone=4,
    ):
        self.root = Path(root).resolve()
        self.workspace = Path(workspace).resolve()
        self.output = Path(output).resolve()
        self.output.relative_to((self.root / "research_runs/rsi").resolve())
        self.allowed_plugins = frozenset(allowed_plugins)
        if not self.allowed_plugins <= {p["id"] for p in list_case_plugins()}:
            raise ValueError("unknown allowed plugin")
        self.expected_parameters = json.loads(json.dumps(expected_parameters or {}))
        if not set(self.expected_parameters) <= self.allowed_plugins:
            raise ValueError("Expected parameters reference an unadmitted plugin")
        if (
            type(maximum_backbones) is not int
            or not 1 <= maximum_backbones <= 1000
            or type(maximum_sequences_per_backbone) is not int
            or not 1 <= maximum_sequences_per_backbone <= 20
        ):
            raise ValueError("Invalid predeclared antibody design limits")
        self.maximum_backbones = maximum_backbones
        self.maximum_sequences_per_backbone = maximum_sequences_per_backbone
        self.max_jobs = max_jobs
        self.timeout_s = timeout_s
        self.jobs = {}
        self.tasks = {}
        self.seen = set()
        self.rpc = self.workspace / "case_rpc"
        self.rpc.mkdir(parents=True, exist_ok=True)
        self.output.mkdir(parents=True, exist_ok=False)
        self.loop_task = None

    def persist(self):
        atomic_json(self.output / "jobs.json", {"jobs": self.jobs})

    def catalog(self):
        return {
            "plugins": [p for p in list_case_plugins() if p["id"] in self.allowed_plugins],
            "maximum_jobs": self.max_jobs,
            "maximum_backbones": self.maximum_backbones,
            "maximum_sequences_per_backbone": self.maximum_sequences_per_backbone,
            "timeout_s": self.timeout_s,
            "maximum_alde_seeds": 70,
            "jobs": [self.status(jid) for jid in self.jobs],
        }

    def status(self, job_id):
        if job_id not in self.jobs:
            raise ValueError("unknown job")
        job = self.jobs[job_id]
        result = {k: v for k, v in job.items() if k not in {"argv", "output", "parameters_hash"}}
        if job["plugin_id"] == "clean.training.v1":
            from agent.rsi.cases.clean_training_plugin import public_progress

            result["progress"] = public_progress(Path(job["output"]))
        return result

    def submit(self, plugin_id, parameters, idempotency_key):
        if plugin_id not in self.allowed_plugins:
            raise ValueError("plugin outside campaign")
        if not isinstance(parameters, dict):
            raise ValueError("Job parameters must be a JSON object")
        if not isinstance(idempotency_key, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,256}", idempotency_key):
            raise ValueError("Idempotency key must contain 1..256 ASCII letters, digits, underscores or hyphens")
        backbones, sequences = antibody_allocation(plugin_id, parameters)
        if plugin_id.startswith("rfantibody") and (
            type(backbones) is not int
            or not 1 <= backbones <= self.maximum_backbones
            or type(sequences) is not int
            or not 1 <= sequences <= self.maximum_sequences_per_backbone
        ):
            raise ValueError("campaign design budget exceeded")
        parameters = json.loads(json.dumps(parameters))
        if plugin_id in self.expected_parameters and digest(parameters) != digest(self.expected_parameters[plugin_id]):
            raise ValueError("Request differs from the predeclared allocation")
        binding = digest({"plugin": plugin_id, "parameters": parameters})
        job_id = "case_" + digest({"key": idempotency_key})[:20]
        if job_id in self.jobs:
            if self.jobs[job_id]["parameters_hash"] != binding:
                raise ValueError("idempotency key reused for different request")
            return self.status(job_id)
        if len(self.jobs) >= self.max_jobs:
            raise ValueError("job budget exhausted")
        out = self.output / job_id
        argv = invocation(plugin_id, parameters, root=self.root, output=out, controller_python=Path(sys.executable))
        self.jobs[job_id] = {
            "job_id": job_id,
            "plugin_id": plugin_id,
            "parameters": parameters,
            "parameters_hash": binding,
            "status": "queued",
            "output": str(out),
            "argv": argv,
            "entry_sha256": sha256_file(Path(argv[2])),
            "resource": resource_for(plugin_id, parameters),
        }
        self.persist()
        self.tasks[job_id] = asyncio.create_task(self.execute(job_id))
        return self.status(job_id)

    async def execute(self, job_id):
        from backend.web_v2.job_queue import resource_slot
        try:
            async with resource_slot(self.jobs[job_id]['resource'], kind='rsi-case'):
                await self._execute_admitted(job_id)
        except asyncio.CancelledError:
            self.jobs[job_id]['status']='cancelled';self.persist();raise
        except Exception as exc:
            self.jobs[job_id].update(status='failed',error_type=type(exc).__name__,error=str(exc)[:500]);self.persist()

    async def _execute_admitted(self, job_id):
        job = self.jobs[job_id]
        log = None
        proc = None
        lock = None
        started = time.monotonic()
        try:
            async with asyncio.timeout(self.timeout_s):
                if job["resource"] == "gpu":
                    lock = (self.root / "research_runs/rsi/.case_gpu.lock").open("a")
                    while True:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            break
                        except BlockingIOError:
                            await asyncio.sleep(0.25)
                if sha256_file(Path(job["argv"][2])) != job["entry_sha256"]:
                    raise ValueError("plugin entry changed while queued")
                job["status"] = "running"
                self.persist()
                home = self.output / "job_home"
                home.mkdir(exist_ok=True)
                env = {
                    "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                    "HOME": str(home),
                    "PYTHONPATH": str(self.root / "src"),
                    "LANG": "C.UTF-8",
                    "CUDA_VISIBLE_DEVICES": "0" if job["resource"] == "gpu" else "",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                }
                if job["plugin_id"] == "clean.training.v1":
                    # The wrapper also replays numerical evidence; use the same
                    # deterministic runtime settings as its training stages.
                    from agent.rsi.cases.clean_training_plugin import environment

                    env = environment(self.root, Path(job["output"]))
                if lock is not None:
                    env["VENUS_RSI_CASE_LOCK_FD"] = str(lock.fileno())
                log = (self.output / (job_id + ".log")).open("wb")
                proc = await asyncio.create_subprocess_exec(
                    *job["argv"],
                    cwd=self.root,
                    env=env,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(lock.fileno(),) if lock else (),
                )
                code = await proc.wait()
                job["exit_code"] = code
                report_path = Path(job["output"]) / "report.json"
                if code != 0 or not report_path.is_file() or report_path.is_symlink():
                    raise ValueError("plugin failed or report missing")
                report = json.loads(report_path.read_text())
                if report.get("status") != "completed":
                    raise ValueError("plugin did not complete domain assessment")
                if job["plugin_id"] == "clean.training.v1":
                    from agent.rsi.cases.clean_training_plugin import verify_artifacts_async

                    if job["parameters"] != {}:
                        raise ValueError("CLEAN training allocation changed")
                    await verify_artifacts_async(report, Path(job["output"]), self.root)
                    evidence = canonical_claims(job["plugin_id"], report)
                else:
                    evidence = verify_job_report(self.root, job, report)
                job.update(status="completed", evidence=evidence, report_sha256=sha256_file(report_path))
        except TimeoutError:
            job["status"] = "timed_out"
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            raise
        except Exception as exc:
            job.update(status="failed", error_type=type(exc).__name__, error=str(exc)[:500])
        finally:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    await asyncio.wait_for(proc.wait(), 3)
                except (TimeoutError, ProcessLookupError):
                    pass
                finally:
                    # A descendant may ignore TERM after its parent has exited.
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await proc.wait()
            if log:
                log.close()
            if lock:
                lock.close()
            job["wall_seconds"] = round(time.monotonic() - started, 3)
            self.persist()

    async def serve(self):
        while True:
            for path in sorted(self.rpc.glob("*.request.json")):
                if path.name in self.seen:
                    continue
                self.seen.add(path.name)
                rid = path.name.removesuffix(".request.json")
                if not re.fullmatch(r"[0-9a-f]{32}", rid) or path.is_symlink() or path.stat().st_size > 10000:
                    continue
                try:
                    request = json.loads(path.read_text())
                    if set(request) - {"action", "plugin_id", "parameters", "idempotency_key", "job_id"}:
                        raise ValueError("undeclared request fields")
                    if request["action"] == "catalog":
                        value = self.catalog()
                    elif request["action"] == "submit":
                        value = self.submit(request["plugin_id"], request["parameters"], request["idempotency_key"])
                    elif request["action"] == "status":
                        value = self.status(request["job_id"])
                    else:
                        raise ValueError("unknown action")
                    response = {"success": True, **value}
                except Exception as exc:
                    response = {"success": False, "error_type": type(exc).__name__, "error": str(exc)[:500]}
                destination = self.rpc / (rid + ".response.json")
                if destination.exists() or destination.is_symlink():
                    raise ValueError("response identity already exists")
                atomic_json(destination, response)
            await asyncio.sleep(0.1)

    async def __aenter__(self):
        self.loop_task = asyncio.create_task(self.serve())
        return self

    async def wait_jobs(self):
        if self.tasks:
            await asyncio.gather(*self.tasks.values())
        if self.loop_task.done():
            self.loop_task.result()

    async def __aexit__(self, *_):
        for task in [self.loop_task, *self.tasks.values()]:
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *[t for t in [self.loop_task, *self.tasks.values()] if t is not None], return_exceptions=True
        )


def verify_job_report(root, job, report):
    """Recheck domain artifacts at execution, registration and reuse assessment."""
    if report.get("status") != "completed":
        raise ValueError("Domain assessment is incomplete")
    if job["plugin_id"] == "rfantibody.benchmark.v1":
        from agent.rsi.cases.rfantibody_benchmark import verify_benchmark_artifacts

        verify_benchmark_artifacts(root, Path(job["output"]), report, job["parameters"])
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"].startswith("rfantibody"):
        if (
            report.get("computational_chain_completed") is not True
            or report.get("quality_audit", {}).get("sequence_preservation_verified") is not True
        ):
            raise ValueError("antibody domain evidence incomplete")
        if job["plugin_id"] == "rfantibody.targets.v1":
            from agent.rsi.cases.rfantibody_evidence import validate_target_report, verify_target_artifacts

            validate_target_report(report, job["parameters"])
            verify_target_artifacts(report, Path(job["output"]))
        evidence = {
            "quality_audit": report["quality_audit"],
            "full_paper_reproduced": False,
            "binding_verified": False,
        }
    elif job["plugin_id"] == "catapro.training.v1":
        from agent.rsi.cases.catapro_training_evidence import verify_artifacts

        if report.get("operation") != job["parameters"]["operation"] or report.get("requested_campaign") != job[
            "parameters"
        ].get("campaign"):
            raise ValueError("CataPro training allocation changed")
        verify_artifacts(report, Path(job["output"]), root)
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "catapro.full_cv.v1":
        from agent.rsi.cases.catapro_evidence import COMMIT, verify_artifacts

        if report.get("operation") != job["parameters"]["operation"] or report.get("requested_campaign") != job[
            "parameters"
        ].get("campaign"):
            raise ValueError("CataPro allocation changed")
        source = root / "research_runs/rsi/case_assets/CataPro" / COMMIT
        verify_artifacts(report, Path(job["output"]), source)
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "clean.training.v1":
        from agent.rsi.cases.clean_training_plugin import verify_artifacts

        if job["parameters"] != {}:
            raise ValueError("CLEAN training allocation changed")
        verify_artifacts(report, Path(job["output"]), root)
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "clean.annotation.v1":
        from agent.rsi.cases.clean_annotation import verify_artifacts

        if job["parameters"] != {}:
            raise ValueError("CLEAN allocation changed")
        verify_artifacts(report, Path(job["output"]), root / "research_runs/rsi/case_assets/CLEAN")
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "nucb.raw_analysis.v1":
        from agent.rsi.cases.nucb_reproduction import verify_artifacts

        if job["parameters"] != {}:
            raise ValueError("NucB allocation changed")
        verify_artifacts(report, Path(job["output"]), root)
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "htfunclib.epinnet.v1":
        from agent.rsi.cases.htfunclib_epinnet import verify_artifacts

        source = root / "research_runs/rsi/case_assets/htFuncLib/inspection_api/htFuncLib_notebook"
        verify_artifacts(report, Path(job["output"]), source)
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "htfunclib.rf.v1":
        from agent.rsi.cases.htfunclib_case import verify_artifacts

        source = root / "research_runs/rsi/case_assets/htFuncLib/inspection_api/random_forest"
        verify_artifacts(report, Path(job["output"]), source)
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "phl7.hrex_validation.v1":
        from agent.rsi.cases.phl7_hrex import verify_artifacts

        if report.get("compute_backend") != job["parameters"].get("backend", "cpu"):
            raise ValueError("HREX backend changed")
        verify_artifacts(report, Path(job["output"]))
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "phl7.opes_analysis.v1":
        from agent.rsi.cases.phl7_case import verify_artifacts

        verify_artifacts(report, Path(job["output"]))
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "mcba.multisubstrate.v1":
        if report.get("operation") != job["parameters"]["operation"] or report.get("requested_campaign") != job[
            "parameters"
        ].get("campaign"):
            raise ValueError("McbA request identity changed")
        audit_path = Path(job["output"]) / "audit/audit.json"
        if not audit_path.is_file() or sha256_file(audit_path) != report.get("audit_report_sha256"):
            raise ValueError("McbA independent audit missing or changed")
        if json.loads(audit_path.read_text()) != report.get("audit_evidence"):
            raise ValueError("McbA audit projection differs from its source")
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "alde.onehot_dnn.v1":
        if report.get("operation") != job["parameters"]["operation"] or report.get("requested_campaign") != job[
            "parameters"
        ].get("campaign"):
            raise ValueError("ALDE request identity changed")
        if report["operation"] == "reproduce" and (
            report.get("seeds") != job["parameters"].get("seeds", 5)
            or report.get("acquisitions") != sorted(job["parameters"].get("acquisitions", ["TS"]))
        ):
            raise ValueError("ALDE allocation changed")
        audit_path = Path(job["output"]) / "audit/report.json"
        if not audit_path.is_file() or sha256_file(audit_path) != report.get("audit_report_sha256"):
            raise ValueError("ALDE independent audit missing or changed")
        evidence = canonical_claims(job["plugin_id"], report)
    elif job["plugin_id"] == "turbopetase.v1":
        comparisons = report.get("comparisons", [])
        original = [x for x in comparisons if x.get("indexing") == "author_notebook"]
        if len(original) != len(job["parameters"].get("models", ["PF01083", "PTM311", "pretrained-model"])) or any(
            x.get("independent_author_function_max_abs_error", 1) > 1e-5 for x in original
        ):
            raise ValueError("PET checkpoint evidence incomplete")
        evidence = {
            "checkpoints": [x["checkpoint"] for x in original],
            "author_inference_replayed": True,
            "measurement_rows": report.get("measurement_rows"),
            "duplicates": report.get("duplicate_mutations"),
            "scope": report.get("claim_scope"),
            "missing_for_complete_paper": report.get("missing_for_complete_paper"),
        }
    else:
        raise ValueError("No domain evidence validator registered")
    return evidence
