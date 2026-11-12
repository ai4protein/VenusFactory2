"""Specific interface guidance for a failed isolated fit; never broaden its mounts."""
from __future__ import annotations

import ast
import re
from pathlib import PurePosixPath


def _workspace_relative(value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        return None
    if value.startswith("/workspace/"):
        value = value[len("/workspace/"):]
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        return None
    return path.as_posix()


def training_data_diagnostic(details, data, *, stage):
    """Match the final Python cause to this prepared table, not arbitrary files.

    preparation.source_file is the same path exposed as training_data.file by
    protected_data_access. The fitting sandbox receives training.json instead;
    this diagnostic changes neither the candidate nor available data.
    """
    if stage != "fitting" or not isinstance(data, dict):
        return None
    preparation = data.get("preparation")
    if not isinstance(preparation, dict):
        return None
    expected = _workspace_relative(preparation.get("source_file"))
    if expected is None:
        return None
    causes = re.findall(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):.*$",
                        str(details), re.MULTILINE)
    if not causes:
        return None
    match = re.fullmatch(r"FileNotFoundError: \[Errno 2\] No such file or directory: (.+)",
                         causes[-1])
    if not match:
        return None
    try:
        missing = ast.literal_eval(match[1])
    except (ValueError, SyntaxError, RecursionError):
        return None
    if _workspace_relative(missing) != expected:
        return None
    return {
        "code": "prepared_training_file_not_mounted",
        "data_access": "training_argument",
        "hint": "The prepared workspace table is not mounted during fitting. "
                "fit(training, seed) receives training['train']; each row has inputs and target. "
                "Use that argument for fitting.",
    }
