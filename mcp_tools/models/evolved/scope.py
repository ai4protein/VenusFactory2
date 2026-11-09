"""Bind short prediction references to immutable, entry-specific model scopes."""
from __future__ import annotations

import re
from agent.harness.contracts import digest


class ModelScopeError(ValueError):
    """Public scope feedback containing no caller values or internal evidence."""


def scope_reference(entry_id, scope):
    return digest({"schema": "evolved_model_scope.v1", "entry_id": entry_id, "scope": scope})


def bind_scope(library, entry_id, scope=None, scope_ref=None):
    # Argument shape and required alternatives belong to PredictionArguments.
    expected = library.inspect_publication(entry_id)["key"]["scope"]
    if scope_ref is not None and scope_ref != scope_reference(entry_id, expected):
        raise ModelScopeError("Scope reference does not match this model. Read describe_evolved_model again.")
    if scope is not None and scope != expected:
        missing = [key for key in expected if key not in scope and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", key)]
        detail = " Missing fields: " + ", ".join(missing[:8]) + "." if missing else ""
        raise ModelScopeError("Scope does not match this model." + detail
                              + " Use scope_ref from describe_evolved_model to retain the complete scope.")
    # References use the canonical scope; legacy inline calls retain their supplied representation.
    return expected if scope_ref is not None else scope
