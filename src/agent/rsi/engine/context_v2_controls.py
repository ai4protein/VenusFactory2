"""Version 2 input validation for the shared fixed numerical controls."""

from __future__ import annotations

if __package__:
    from . import context_controls
    from .generated_context_v2 import validate_rows
else:  # Frozen standalone capability bundle.
    import context_controls
    from generated_context_v2 import validate_rows

# Preserve the existing numerical helper interface for callers and old imports.
METHODS = context_controls.METHODS
THERMAL_METHODS = context_controls.THERMAL_METHODS
thermal_methods = context_controls.thermal_methods
estimators = context_controls.estimators
sigmoid = context_controls.sigmoid
thermal_predictions = context_controls.thermal_predictions


def score_rows(training, validation, scope, *, timings=True):
    validate_rows(training, scope, labels=True)
    validate_rows(validation, scope, labels=True)
    return context_controls.score_validated_rows(training, validation, scope, timings=timings)
