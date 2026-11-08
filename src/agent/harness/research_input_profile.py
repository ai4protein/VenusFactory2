"""Bounded training-only representation summaries; no observations or targets."""
from __future__ import annotations

import json
import math

MAX_FIELDS = 16
MAX_BYTES = 6000


def training_input_profile(rows, columns):
    """Accept training rows alone so held-out observations cannot affect the profile."""
    features = []
    for name in columns[:MAX_FIELDS]:
        values = [row["inputs"][name] for row in rows]
        nonempty = [value for value in values if value.strip()]
        numeric = 0
        for value in nonempty:
            try:
                numeric += math.isfinite(float(value))
            except (ValueError, OverflowError):
                pass
        entry = {
            "name": name, "storage": "string", "empty_rows": len(values) - len(nonempty),
            "finite_numeric_rows": numeric, "distinct_values": len(set(values)),
            "length_range": [min(map(len, values), default=0), max(map(len, values), default=0)],
        }
        if len(json.dumps([*features, entry], ensure_ascii=False).encode("utf-8")) > MAX_BYTES:
            break
        features.append(entry)
    return {
        "scope": "training_only", "rows": len(rows), "feature_count": len(columns),
        "features": features, "omitted_features": len(columns) - len(features),
        "interpretation": "CSV fields remain strings. Counts describe stored values, not biological meaning. "
                          "Convert numerical fields or learn an appropriate encoding from training rows; "
                          "retain that transformation for prediction.",
    }
