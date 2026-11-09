"""Fixed controls for measured protein-condition response tasks."""

from pathlib import Path

import numpy as np
import scipy
import sklearn

from agent.rsi.datasets.model_tasks import load_task

from . import context_controls
from . import reference_thermal_baselines as reference_thermal
from .context_controls import (
    METHODS as METHODS,
)
from .context_controls import (
    THERMAL_METHODS as THERMAL_METHODS,
)
from .context_controls import (
    estimators as estimators,
)
from .context_controls import (
    sigmoid as sigmoid,
)
from .context_controls import (
    thermal_methods as thermal_methods,
)
from .context_controls import (
    thermal_predictions as thermal_predictions,
)
from .protein_model import digest
from .storage import sha256_file

VERSION = "context_classical_regression.v2"


def run_controls(task, *, score_partition="validation"):
    if score_partition not in {"validation", "future"}:
        raise ValueError("Controls score validation or future observations; fitting uses training only")
    if task["plugin_id"] not in {"generated_context.v1", "generated_context.v2", "generated_context.v3"}:
        raise ValueError("Context response interface required")
    data = load_task(task)
    training, validation = data["training"], data[score_partition]
    controls = context_controls
    if task["plugin_id"] == "generated_context.v2":
        from . import context_v2_controls as controls
    elif task["plugin_id"] == "generated_context.v3":
        from . import context_v3_controls as controls
    outcomes = controls.score_rows(training, validation, task["scope"])
    return {
        "schema": VERSION,
        "task_id": task["id"],
        "task_sha256": digest(task),
        "implementation_sha256": sha256_file(Path(__file__)),
        "numeric_implementation_sha256": sha256_file(Path(controls.__file__)),
        "reference_implementation_sha256": sha256_file(Path(reference_thermal.__file__)),
        "fit_partition": "training",
        "score_partition": score_partition,
        "future_scored": score_partition == "future",
        "hyperparameters": "fixed in versioned implementation; no validation selection",
        "representation": task["scope"]["covariates"],
        "thermal_reference": "decreasing isotonic and four-parameter sigmoid fits, conditional on other recorded covariates; no author-pipeline equivalence claim",
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "rows": {p: len(data[p]) for p in ("training", score_partition)},
        "groups": {p: len({r["group"] for r in data[p]}) for p in ("training", score_partition)},
        "outcomes": outcomes,
        "domain_baseline_comparison_complete": False,
        "new_autonomous_models": 0,
    }
