"""Trusted JSON ABI worker, executed only inside the networkless code sandbox."""

import importlib.util
import json
import random
import resource
import traceback
from pathlib import Path

resource.setrlimit(resource.RLIMIT_CPU, (90, 90))
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))


def serialize(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("model state must be JSON serializable")


request = json.loads(Path("/input/request.json").read_text())
seed = int(request.get("seed", 0))
random.seed(seed)
import numpy as np

np.random.seed(seed)
# Reaching this marker distinguishes candidate execution from worker startup errors.
Path("/output/worker_started").write_text("candidate_execution_started\n")
try:
    spec = importlib.util.spec_from_file_location("generated_protein_model", "/candidate/model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    input_kind = request.get("input_kind", "sequences")
    if input_kind not in {"sequences", "features"}:
        raise ValueError("unsupported model input kind")
    inputs = request[input_kind]
    if request["operation"] == "fit":
        result = {"model": module.fit(inputs, request["targets"], seed)}
    elif request["operation"] == "predict":
        result = {"predictions": module.predict(request["model"], inputs)}
    else:
        raise ValueError("unsupported operation")
    encoded = json.dumps(result, default=serialize, allow_nan=False)
except Exception as exc:
    encoded = json.dumps(
        {
            "candidate_error": {
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
                "traceback": traceback.format_exc()[-3000:],
            }
        },
        allow_nan=False,
    )
if request.get("record_runtime"):
    try:
        runtime_spec = importlib.util.spec_from_file_location("numeric_runtime", "/runner/numeric_runtime.py")
        runtime_module = importlib.util.module_from_spec(runtime_spec)
        runtime_spec.loader.exec_module(runtime_module)
        Path("/output/runtime.json").write_text(json.dumps(runtime_module.capture_runtime(), allow_nan=False))
    except Exception as exc:
        Path("/output/runtime_failed").write_text(type(exc).__name__)
        raise
Path("/output/result.json").write_text(encoded)
