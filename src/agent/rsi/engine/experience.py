"""Host-verified, append-only model experience with controlled public projections.

Experience transfers implementations, never fitted weights or assessment labels.
A published donor is evidence about its original scope only. A new task must
refit, independently assess and publish its own candidate.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

try:
    from .code_sandbox import guard_source
    from .protein_model import digest
except ImportError:  # immutable MCP bundle
    from code_sandbox import guard_source
    from protein_model import digest

CONDITIONS = {"none", "summary", "source"}
ENTRY_KEYS = {"experience_id", "source_sha256", "donor_scope", "method_summary", "acceptance", "parents"}
LIMITATION = "Donor acceptance is scope-specific. Source is an implementation precedent, not a validated target model; refit and independent target acceptance are required."


def validate_projection(packet):
    if set(packet) != {"schema", "condition", "entries", "limitation"} or packet["schema"] != 1:
        raise ValueError("invalid experience projection")
    condition = packet["condition"]
    if condition not in CONDITIONS or packet["limitation"] != LIMITATION:
        raise ValueError("invalid experience condition")
    if not isinstance(packet["entries"], list) or len(packet["entries"]) > 30:
        raise ValueError("invalid experience list")
    if condition == "none" and packet["entries"]:
        raise ValueError("cold-start projection contains history")
    seen = set()
    for entry in packet["entries"]:
        if set(entry) != ENTRY_KEYS | ({"source_code"} if condition == "source" else set()):
            raise ValueError("unapproved experience fields")
        if entry["experience_id"] in seen:
            raise ValueError("duplicate experience")
        seen.add(entry["experience_id"])
        for name in ("experience_id", "source_sha256"):
            value = entry[name]
            if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
                raise ValueError("invalid experience digest")
        if entry["acceptance"] != {"passed": True, "training_replayed": True}:
            raise ValueError("experience was not independently accepted")
        if not isinstance(entry["method_summary"], str) or not 1 <= len(entry["method_summary"]) <= 6000:
            raise ValueError("invalid method summary")
        scope = entry["donor_scope"]
        base_scope = {"protein", "assay", "target", "unit", "sequence_length"}
        variable_scope = (base_scope - {"sequence_length"}) | {"minimum_sequence_length", "maximum_sequence_length"}
        context_scope = {"protein", "assay", "target", "unit", "context", "covariates", "grouping"}
        if set(scope) not in (
            context_scope,
            context_scope | {"interface_version"},
            base_scope,
            base_scope | {"class_labels"},
            base_scope | {"representation"},
            variable_scope,
        ):
            raise ValueError("invalid donor scope")
        if set(scope) in (context_scope, context_scope | {"interface_version"}):
            try:
                from .context_interfaces import validate_scope
            except ImportError:
                from context_interfaces import validate_scope

            validate_scope(scope)
        if set(scope) == variable_scope:
            try:
                from .generated_sequence import validate_scope
            except ImportError:
                from generated_sequence import validate_scope

            validate_scope(scope)
        if "representation" in scope:
            try:
                from .generated_features import validate_scope
            except ImportError:
                from generated_features import validate_scope

            validate_scope(scope)
        if "class_labels" in scope:
            labels = scope["class_labels"]
            if (
                not isinstance(labels, list)
                or not 2 <= len(labels) <= 20
                or any(not isinstance(x, str) or not x for x in labels)
                or len(set(labels)) != len(labels)
            ):
                raise ValueError("invalid donor class labels")
        if not isinstance(entry["parents"], list) or any(
            not isinstance(p, str) or len(p) != 64 for p in entry["parents"]
        ):
            raise ValueError("invalid parent lineage")
        if (
            condition == "source"
            and hashlib.sha256(entry["source_code"].encode()).hexdigest() != entry["source_sha256"]
        ):
            raise ValueError("experience source changed")
    return packet


class ModelExperienceStore:
    def __init__(self, root: Path):
        from .storage import atomic_json, file_lock

        self.root = root
        (root / "entries").mkdir(parents=True, exist_ok=True)
        with file_lock(root / ".experience.lock"):
            if not (root / "catalog.json").exists():
                atomic_json(root / "catalog.json", {"schema": 1, "entries": []})

    def _catalog(self):
        catalog = json.loads((self.root / "catalog.json").read_text())
        if set(catalog) != {"schema", "entries"} or catalog["schema"] != 1:
            raise ValueError("corrupt experience catalog")
        if len(catalog["entries"]) != len(set(catalog["entries"])):
            raise ValueError("duplicate experience catalog entry")
        return catalog

    def _entry(self, identity):
        if not isinstance(identity, str) or len(identity) != 64 or set(identity) - set("0123456789abcdef"):
            raise ValueError("invalid experience identity")
        path = self.root / "entries" / (identity + ".json")
        if path.is_symlink():
            raise ValueError("symlink experience rejected")
        entry = json.loads(path.read_text())
        if digest(entry) != identity:
            raise ValueError("experience content changed")
        return entry

    def publish(self, lifecycle, model_id, *, summary, parents=()):
        from .storage import atomic_json, file_lock

        if not isinstance(summary, str) or not 1 <= len(summary) <= 6000:
            raise ValueError("a bounded method summary is required")
        # export() verifies lifecycle publication, frozen artifact and assessment.
        with tempfile.TemporaryDirectory(prefix="vf-experience-") as raw:
            packet = lifecycle.export(model_id, Path(raw) / "export")
        model, assessment = packet["model"], packet["assessment"]
        if model.get("development_type") != "agent_generated_model_code" or not assessment.get("training_replayed"):
            raise ValueError("experience requires independently replayed generated source")
        review = guard_source(model["recipe"]["source_code"])
        if review["source_sha256"] != model["source_sha256"]:
            raise ValueError("source provenance mismatch")
        entry = {
            "schema": 1,
            "model_id": model_id,
            "source_sha256": model["source_sha256"],
            "source_code": model["recipe"]["source_code"],
            "donor_scope": model["scope"],
            "method_summary": summary,
            "acceptance": {"passed": True, "training_replayed": True},
            "assessment_sha256": digest(assessment),
            "model_sha256": packet["model_sha256"],
            "parents": list(parents),
        }
        identity = digest(entry)
        with file_lock(self.root / ".experience.lock"):
            catalog = self._catalog()
            if not set(parents) <= set(catalog["entries"]):
                raise ValueError("unknown experience parent")
            for parent in parents:
                self._entry(parent)
            if identity not in catalog["entries"]:
                atomic_json(self.root / "entries" / (identity + ".json"), entry)
                catalog["entries"].append(identity)
                atomic_json(self.root / "catalog.json", catalog)
            else:
                self._entry(identity)
        return identity

    def project(self, condition, identities=None):
        if condition not in CONDITIONS:
            raise ValueError("unknown experience condition")
        catalog = self._catalog()
        identities = catalog["entries"] if identities is None else list(identities)
        if not set(identities) <= set(catalog["entries"]):
            raise ValueError("experience selection is not published")
        entries = []
        for identity in identities:
            original = self._entry(identity)
            if condition == "none":
                continue
            entry = {key: original[key] for key in ENTRY_KEYS - {"experience_id"}}
            entry["experience_id"] = identity
            if condition == "source":
                entry["source_code"] = original["source_code"]
            entries.append(entry)
        return validate_projection({"schema": 1, "condition": condition, "entries": entries, "limitation": LIMITATION})
