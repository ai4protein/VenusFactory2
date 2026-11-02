"""Explicit study-disjoint admission of portable contextual fit/predict source.

Study and sample identities are host-reviewed provenance, not LLM assertions.
A different DOI is a checked identifier boundary, not proof of biological
independence or a license to transfer measured labels or fitted weights.
"""

from __future__ import annotations

import re
from urllib.parse import unquote

from agent.rsi import context_interfaces, model_tasks

POLICY = "context_study_disjoint.v1"
SOURCE_ADAPTATION = "context_numeric_source_adaptation.v1"
RULE = (
    "All direct and ancestral donors require source-bound contextual study/group evidence "
    "and a canonical study DOI distinct from the receiving study; every recipient is refitted "
    "and independently assessed. DOI identity does not establish biological independence."
)


def validate_declaration(value, *, plugin_id=None):
    if "adaptation_policy" in value:
        if value["adaptation_policy"] != SOURCE_ADAPTATION:
            raise ValueError("Unsupported contextual source adaptation policy")
        if value.get("overlap_policy") != POLICY:
            raise ValueError("Numeric source adaptation requires explicit study-disjoint admission")
    if "overlap_policy" not in value:
        return None
    if value["overlap_policy"] != POLICY:
        raise ValueError("Unsupported contextual source overlap policy")
    if plugin_id is not None and plugin_id not in context_interfaces.PLUGINS:
        raise ValueError("Contextual source overlap policy requires the context model interface")
    return POLICY


def canonical_doi(value):
    if not isinstance(value, str):
        raise ValueError("Contextual source transfer requires a study DOI")
    value = unquote(value.strip()).casefold()
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)
    if len(value) > 256 or re.fullmatch(r"10\.\d{4,9}/[^\s?#]+", value) is None:
        raise ValueError("Contextual source transfer requires an unambiguous study DOI")
    return value


def interface(scope):
    context_interfaces.validate_scope(scope)
    return {
        "assay": scope["assay"],
        "target": scope["target"],
        "unit": scope["unit"],
        "covariates": [{"name": row["name"], "unit": row["unit"]} for row in scope["covariates"]],
    }


def describe_task(task, data):
    if task["plugin_id"] not in context_interfaces.PLUGINS:
        raise ValueError("Contextual source ancestry contains a non-context task")
    evidence = model_tasks.context_group_evidence(task, data)
    study = canonical_doi(evidence["study_doi"])
    groups = {phase: sorted({row["group"] for row in data[phase]}) for phase in model_tasks.PHASES}
    # Existing adapters use DOI-qualified sample IDs. A changed study label must
    # not silently relabel such IDs; unqualified group names are study-local.
    for group in (group for values in groups.values() for group in values):
        if group.casefold().startswith("10.") and ":" in group:
            qualified_study = canonical_doi(group.rsplit(":", 1)[0])
            if qualified_study != study:
                raise ValueError("Context sample study differs from its grouping evidence")
    return {
        "study_doi": study,
        "grouping_level": evidence["level"],
        "grouping_evidence_sha256": task["scope"]["grouping"]["evidence_sha256"],
        "source_evidence_sha256": sorted({row["sha256"] for row in evidence["sources"]}),
        "sample_groups": groups,
        "interface": interface(task["scope"]),
    }
