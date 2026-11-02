"""Host-controlled candidate -> independently assessed -> published -> reused lifecycle.

The registry and evaluation labels stay outside the agent mount. An MCP client
can propose artifacts; it cannot call this controller or self-report acceptance.
Portable exports are read-only snapshots, with hashes in the capability bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.research_modes import get_research_mode

from .code_sandbox import capture_code_runs
from .model_plugins import get_model_plugin
from .numeric_runtime import capture_runtime
from .numerical_evidence import archive_assessment, verify_assessment
from .protein_model import digest
from .protein_references import audit_source_references
from .storage import atomic_json, file_lock
from .versions import VersionStore


class ModelLifecycle:
    def __init__(self, root: Path, policy: dict):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
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
        ):
            raise ValueError("invalid predeclared assessment policy")
        if (
            policy["max_candidates"] != 1
            or type(policy["minimum_skill"]) not in (float, int)
            or not 0 <= policy["minimum_skill"] < 1
        ):
            raise ValueError("pilot permits one candidate and a finite nonnegative skill threshold")
        plugin = get_model_plugin(policy["plugin_id"])
        if plugin.source_sha256 != policy["plugin_sha256"]:
            raise ValueError("plugin implementation changed after policy freeze")
        self.policy = policy
        self.versions = VersionStore(root / "artifacts")
        with file_lock(root / ".controller.lock"):
            dest = root / "policy.json"
            if dest.exists():
                if json.loads(dest.read_text()) != policy:
                    raise ValueError("assessment policy is immutable")
            else:
                atomic_json(dest, policy)
            if not (root / "state.json").exists():
                atomic_json(root / "state.json", {"schema": 1, "candidates": {}, "published": [], "events": []})

    def _state(self):
        state = json.loads((self.root / "state.json").read_text())
        if (
            state.get("schema") != 1
            or not isinstance(state.get("candidates"), dict)
            or not isinstance(state.get("published"), list)
            or not isinstance(state.get("events"), list)
        ):
            raise ValueError("corrupt lifecycle history")
        return state

    def submit(self, artifact: Path, *, session_id: str, mode: str) -> str:
        get_research_mode(mode)
        model = json.loads(artifact.read_text())
        if model.get("plugin_id") != self.policy["plugin_id"] or model.get("scope") != self.policy["scope"]:
            raise ValueError("candidate scope/plugin mismatch")
        if artifact.is_symlink():
            raise ValueError("candidate artifact must not be a symlink")
        with file_lock(self.root / ".controller.lock"):
            state = self._state()
            if len(state["candidates"]) >= self.policy["max_candidates"]:
                raise ValueError("candidate budget exhausted")
            staged = self.root / "submission"
            staged.mkdir(exist_ok=False)
            atomic_json(staged / "model.json", model)
            version = self.versions.snapshot(
                staged, metadata={"session_id": session_id, "mode": mode, "policy_sha256": digest(self.policy)}
            )
            vid = version["version_id"]
            state["candidates"][vid] = {"status": "candidate", "session_id": session_id, "mode": mode}
            state["events"].append({"event": "candidate", "model_id": vid, "session_id": session_id, "mode": mode})
            atomic_json(self.root / "state.json", state)
            return vid

    def _model(self, model_id):
        self.versions.verify(model_id)
        return json.loads((self.root / "artifacts/versions" / model_id / "model.json").read_text())

    def assess(self, model_id: str, *, training: list[dict], validation: list[dict]) -> dict:
        """Trusted caller supplies frozen data; candidate metrics/acceptance flags are ignored."""
        with file_lock(self.root / ".controller.lock"):
            state = self._state()
            if state["candidates"].get(model_id, {}).get("status") != "candidate":
                raise ValueError("candidate is missing or has already been assessed")
            if (
                digest(training) != self.policy["training_sha256"]
                or digest(validation) != self.policy["validation_sha256"]
            ):
                raise ValueError("frozen assessment data changed")
            model = self._model(model_id)
            plugin = get_model_plugin(self.policy["plugin_id"])
            if plugin.source_sha256 != self.policy["plugin_sha256"]:
                raise ValueError("plugin implementation drift")
            # Replay chosen recipe on trusted training rows; no agent-written weights,
            # provenance or baseline numbers are accepted without verification.
            if plugin.evaluate_with_predictions is None:
                raise ValueError("Model plugin lacks per-row assessment evidence")
            with capture_code_runs() as executions:
                rebuilt = plugin.fit(training, self.policy["scope"], model["recipe"])
                if digest(rebuilt) != digest(model):
                    raise ValueError("candidate does not match independently replayed training")
                metrics, predictions = plugin.evaluate_with_predictions(model, validation, self.policy["scope"])
            evidence = archive_assessment(
                self.root, model_id, model, self.policy, metrics, predictions, validation, executions, capture_runtime()
            )
            passed = metrics[plugin.acceptance_metric] > self.policy["minimum_skill"]
            report = {
                "schema": 1,
                "model_id": model_id,
                "policy_sha256": digest(self.policy),
                "plugin_sha256": plugin.source_sha256,
                "metrics": metrics,
                "acceptance_metric": plugin.acceptance_metric,
                "passed": passed,
                "training_replayed": True,
                "numerical_evidence": evidence,
            }
            if isinstance(model.get("recipe", {}).get("source_code"), str):
                report["source_reference_audit"] = audit_source_references(model["recipe"]["source_code"])
            atomic_json(self.root / f"assessment_{model_id}.json", report)
            state["candidates"][model_id].update(
                status="registered" if passed else "rejected", assessment_sha256=digest(report)
            )
            if passed:
                state["events"].append({"event": "validated", "model_id": model_id})
                state["published"].append(model_id)
            state["events"].append(
                {
                    "event": "registered" if passed else "rejected",
                    "model_id": model_id,
                    "assessment_sha256": digest(report),
                }
            )
            atomic_json(self.root / "state.json", state)
            return report

    def _verified_assessment(self, model_id, state, model):
        report = json.loads((self.root / f"assessment_{model_id}.json").read_text())
        if digest(report) != state["candidates"][model_id]["assessment_sha256"] or not report["passed"]:
            raise ValueError("assessment record was changed")
        verify_assessment(self.root, model_id, model, self.policy, report)
        return report

    def export(self, model_id: str, destination: Path) -> dict:
        state = self._state()
        if model_id not in state["published"] or state["candidates"][model_id]["status"] != "registered":
            raise ValueError("only independently validated models can be exported")
        model = self._model(model_id)
        report = self._verified_assessment(model_id, state, model)
        destination.mkdir(parents=True, exist_ok=False)
        packet = {
            "schema": 1,
            "model_id": model_id,
            "model": model,
            "model_sha256": digest(model),
            "assessment": report,
        }
        atomic_json(destination / "capability.json", packet)
        return packet

    def record_reuse(
        self, model_id: str, *, session_id: str, mode: str, inputs: list[dict], predictions: list[dict], scope: dict
    ) -> dict:
        get_research_mode(mode)
        with file_lock(self.root / ".controller.lock"):
            state = self._state()
            record = state["candidates"].get(model_id, {})
            if model_id not in state["published"] or record.get("status") != "registered":
                raise ValueError("model is not available for reuse")
            if not session_id or session_id == record["session_id"]:
                raise ValueError("reuse requires a distinct session")
            get_model_plugin(self.policy["plugin_id"]).validate_rows(inputs, scope, labels=False)
            model = self._model(model_id)
            self._verified_assessment(model_id, state, model)
            if model["plugin_id"] in {
                "generated_context.v1",
                "generated_context.v2",
                "generated_context.v3",
                "generated_assay.v1",
                "generated_antibody.v1",
            }:
                from .context_interfaces import module_for

                module_for(model["plugin_id"]).require_independent(model, inputs)
            elif set(model["training_sequences"]) & {r["sequence"] for r in inputs}:
                raise ValueError("research reuse requires unseen sequences")
            expected = get_model_plugin(model["plugin_id"]).predict(model, inputs, scope)
            if digest(expected) != digest(predictions):
                raise ValueError("reuse output does not match validated model inference")
            event = {
                "event": "reused",
                "model_id": model_id,
                "session_id": session_id,
                "mode": mode,
                "input_sha256": digest(inputs),
                "prediction_sha256": digest(predictions),
                "rows": len(inputs),
            }
            state["events"].append(event)
            atomic_json(self.root / "state.json", state)
            return event

    def retire(self, model_id: str, reason: str) -> None:
        if not reason.strip():
            raise ValueError("retirement reason required")
        with file_lock(self.root / ".controller.lock"):
            state = self._state()
            if model_id not in state["published"]:
                raise ValueError("model is not registered")
            state["published"].remove(model_id)
            state["candidates"][model_id]["status"] = "retired"
            state["events"].append({"event": "retired", "model_id": model_id, "reason": reason})
            atomic_json(self.root / "state.json", state)
