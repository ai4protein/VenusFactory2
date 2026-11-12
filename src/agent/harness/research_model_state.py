"""Fitted-state transport: host validates bytes; generated runtimes load in isolation."""
from __future__ import annotations

import hashlib
import json
import re

from .contracts import digest
from utils.scoped_files import read_bytes

MAX_STATE_BYTES = 64 * 1024**2
STATE_MANIFEST = "model-state.json"
STATE_SCHEMA = "venusfactory.fitted-model.v1"
FORMATS = {"json": "model.json", "pickle-v5": "model.pkl"}


def validate_manifest(value):
    if not isinstance(value, dict) or value.get("schema") != STATE_SCHEMA:
        raise ValueError("Invalid fitted-state manifest")
    if not isinstance(value.get("format"), str) or value["format"] not in FORMATS:
        raise ValueError("Invalid fitted-state format")
    if value.get("file") != FORMATS[value["format"]]:
        raise ValueError("Invalid fitted-state format or filename")
    if type(value.get("bytes")) is not int or not 0 < value["bytes"] <= MAX_STATE_BYTES:
        raise ValueError("Fitted state exceeds the 64 MB limit or is empty")
    if not isinstance(value.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", value["sha256"]):
        raise ValueError("Invalid fitted-state digest")
    if not isinstance(value.get("python"), str) or not re.fullmatch(r"\d+\.\d+", value["python"]):
        raise ValueError("Invalid fitted-state Python version")
    dependencies = value.get("dependencies")
    if not isinstance(dependencies, dict) or len(dependencies) > 128 or any(
        not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", name)
        or not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+!-]{0,99}", version)
        for name, version in dependencies.items()
    ):
        raise ValueError("Invalid fitted-state dependencies")
    return value


def reject_nonfinite(_value):
    raise ValueError("Nonfinite fitted state")


def read_fitted_state(workspace):
    metadata_raw = read_bytes(workspace, STATE_MANIFEST, max_bytes=32768)
    metadata = validate_manifest(json.loads(metadata_raw))
    raw = read_bytes(workspace, metadata["file"], max_bytes=MAX_STATE_BYTES)
    if len(raw) != metadata["bytes"] or hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
        raise ValueError("Fitted state does not match its manifest")
    if metadata["format"] == "json":
        state = json.loads(raw, parse_constant=reject_nonfinite)
        if not isinstance(state, dict):
            raise ValueError("JSON fitted state must be an object")
        state_hash = digest(state)
    else:
        # Never deserialize an estimator in the API, coordinator or review process.
        state_hash = hashlib.sha256(raw).hexdigest()
    return metadata, {STATE_MANIFEST: metadata_raw, metadata["file"]: raw}, state_hash


RUNTIME_SUFFIX = r"""
import hashlib as _vf_hashlib
import importlib.metadata as _vf_metadata
import io as _vf_io
import json as _vf_json
import pickle as _vf_pickle
import sys as _vf_sys
from pathlib import Path as _vf_Path
_vf_state = None
_vf_formats = {"json": "model.json", "pickle-v5": "model.pkl"}


class _vf_Unpickler(_vf_pickle.Unpickler):
    def find_class(self, module, name):
        # Training imports this exact source under research_algorithm.
        if module == "research_algorithm":
            parts = name.split(".")
            value = globals()[parts[0]]
            for part in parts[1:]:
                value = getattr(value, part)
            return value
        return super().find_class(module, name)


def _vf_save(state):
    if not isinstance(state, dict) and not callable(getattr(state, "predict", None)):
        raise TypeError("fit(training, seed) must return fitted parameter dict or estimator; not text, a list or predictions")
    try:
        raw = _vf_json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        encoding = "json"
    except TypeError:
        raw = _vf_pickle.dumps(state, protocol=5)
        encoding = "pickle-v5"
    if len(raw) > 67108864:
        raise ValueError("Fitted state exceeds the 64 MB limit")
    roots = {name.split(".")[0] for name in _vf_sys.modules}
    aliases = {"sklearn": "scikit-learn", "PIL": "Pillow", "yaml": "PyYAML",
               "Bio": "biopython", "cv2": "opencv-python"}
    dependencies = {}
    for root in sorted(roots - _vf_sys.stdlib_module_names):
        if root.startswith("_"):
            continue
        package = aliases.get(root, root)
        try:
            dependencies[package] = _vf_metadata.version(package)
        except _vf_metadata.PackageNotFoundError:
            continue
    metadata = {"schema": "venusfactory.fitted-model.v1", "format": encoding,
                "file": _vf_formats[encoding], "bytes": len(raw),
                "sha256": _vf_hashlib.sha256(raw).hexdigest(),
                "python": ".".join(map(str, _vf_sys.version_info[:2])),
                "dependencies": dependencies}
    base = _vf_Path(__file__).resolve().parent
    (base / metadata["file"]).write_bytes(raw)
    (base / "model-state.json").write_text(_vf_json.dumps(metadata, sort_keys=True))
    return len(raw)


def run(inputs):
    global _vf_state
    if _vf_state is None:
        base = _vf_Path(__file__).resolve().parent
        metadata = _vf_json.loads((base / "model-state.json").read_text())
        encoding = metadata.get("format")
        if metadata.get("schema") != "venusfactory.fitted-model.v1" or encoding not in _vf_formats:
            raise ValueError("Invalid fitted-state manifest")
        if metadata.get("file") != _vf_formats[encoding]:
            raise ValueError("Invalid fitted-state filename")
        if metadata.get("python") != ".".join(map(str, _vf_sys.version_info[:2])):
            raise ValueError("Use the Python version recorded in model-state.json")
        for package, version in metadata["dependencies"].items():
            if _vf_metadata.version(package) != version:
                raise ValueError("Model dependency version differs: " + package)
        path = base / _vf_formats[encoding]
        if path.stat().st_size > 67108864:
            raise ValueError("Fitted state exceeds the 64 MB limit")
        raw = path.read_bytes()
        if len(raw) != metadata["bytes"] or _vf_hashlib.sha256(raw).hexdigest() != metadata["sha256"]:
            raise ValueError("Fitted state does not match its manifest")
        _vf_state = (_vf_json.loads(raw) if encoding == "json"
                     else _vf_Unpickler(_vf_io.BytesIO(raw)).load())
    result = predict(_vf_state, inputs)
    if not isinstance(result, dict):
        raise TypeError("predict(model, inputs) must return an output object")
    return result
"""

TRAINING_SCRIPT = r"""import importlib.util,json,sys
from pathlib import Path
base=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("research_algorithm",base/"runtime.py")
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
training=json.loads((base/"training.json").read_text())

def fit_failure(exc):
    # Keep full chained stderr as local evidence; expose a compact root cause.
    detail = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
    message = f"{type(exc).__name__}: {detail[:240]}."
    fields = training["train"][0]["inputs"] if training["train"] else {}
    key = exc.args[0] if isinstance(exc, KeyError) and exc.args else None
    if isinstance(key, str) and key in fields:
        field = repr(key)[:120]
        return message + (
            " Input contract: in fit, iterate training['train']; "
            f"read row['inputs'][{field}] and row['target']. "
            f"In predict, read inputs[{field}] directly; it has no nested 'inputs' key."
        )
    lower = detail.lower()
    if (isinstance(exc, (TypeError, ValueError))
            and lower.startswith("invalid file path or buffer object type:")
            and "<class 'dict'>" in lower):
        return message + (
            " Input contract: fit receives an in-memory training dictionary, not a CSV path or file buffer. "
            "Do not pass training to pandas.read_csv. Use rows=training['train']; "
            "X=[row['inputs'] for row in rows] and y=[row['target'] for row in rows]. "
            "Build and fit the representation from X, then retain that transformation for predict."
        )
    target_mapping_error = (
        isinstance(key, str) and key not in {"train", "inputs", "target", "labels"}
        or "uniformly strings or numbers" in lower and "dict" in lower
        or "unhashable type" in lower and "dict" in lower
    )
    if training.get("task") == "multilabel" and target_mapping_error:
        return message + (
            " Multilabel target contract: row['target'] already maps each label name to 0, 1 or None; "
            "the original CSV target-column name is not another nested key. "
            "For each label in training['labels'], use row['target'][label] and exclude None cells from its loss. "
            "LabelEncoder is for scalar classes, not these target dictionaries. "
            "predict returns independent probabilities for every declared label."
        )
    if "all arrays must be of the same length" in lower:
        return message + (
            " Tabular contract: build the feature table from row['inputs'] in training['train'], "
            "and targets from row['target']; training itself also contains task/strategy metadata. "
            "Every feature column and target vector must describe the same samples."
        )
    if any(text in lower for text in ("2d array", "1d array", "could not convert string",
                                       "dtype=")):
        return message + (
            " Feature contract: a numeric estimator needs a two-dimensional feature matrix "
            "(one row per sample). Encode raw strings/dicts in your pipeline first. "
            "Column-name selectors require actual tabular columns, not a list of strings. "
            "Retain the fitted transformation and apply it to the single prediction input."
        )
    if isinstance(exc, KeyError) or "string indices" in lower or "list indices" in lower:
        return message + (
            " Input contract: training is a dict; rows=training['train'] is the sample list. "
            "Each row has row['inputs'] and row['target']. predict receives the inputs dict alone. "
            "Before numeric fitting, encode sequence/categorical features and convert numeric strings."
        )
    return message

try:
    state=module.fit(training,__VF_SEED__)
except FileNotFoundError as exc:
    raise ValueError(
        f"FileNotFoundError: {str(exc)[:240]}. "
        "Fitting runs in a fresh training workspace; original session files are not mounted. "
        "Use the supplied training['train'] sample list, row['inputs'] and row['target'], "
        "instead of reopening the uploaded CSV. Keep validation and test data outside fitting."
    ) from exc
except (TypeError, ValueError, KeyError, IndexError) as exc:
    raise ValueError(fit_failure(exc)) from exc
size=module._vf_save(state)
print(json.dumps({"trained_rows":len(training["train"]),"state_bytes":size}))
"""


def training_script(seed):
    return TRAINING_SCRIPT.replace("__VF_SEED__", str(seed))
