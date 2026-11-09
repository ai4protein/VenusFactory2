"""Every model API emits browser-compatible JSON using the same transport."""
import json
import math

import pytest

from backend.web_v2.sse_events import encode_event


def decode(frame):
    event, data = frame.removesuffix("\n\n").split("\n")
    assert event == "event: metrics"
    assert data.startswith("data: ")
    def reject_constant(value):
        raise AssertionError("Non-JSON numeric constant: " + value)
    return json.loads(data[6:], parse_constant=reject_constant)


@pytest.mark.parametrize("ascii_only", [True, False])
def test_finite_nested_payload_and_unicode_roundtrip(ascii_only):
    value = {"message": "蛋白\nmodel", "values": [0.0, -1.5, None], "nested": {"ok": True}}
    frame = encode_event("metrics", value, ensure_ascii=ascii_only)
    assert frame.endswith("\n\n")
    assert decode(frame) == value
    assert ("蛋白" not in frame) == ascii_only


def test_nonfinite_values_become_gaps_without_mutating_source():
    value = {"curve": [float("nan"), float("inf"), float("-inf"), 0.3],
             "nested": {"r2": float("nan")}, "tuple": (float("inf"), 2)}
    assert decode(encode_event("metrics", value)) == {
        "curve": [None, None, None, 0.3], "nested": {"r2": None}, "tuple": [None, 2]
    }
    assert math.isnan(value["curve"][0]) and math.isinf(value["curve"][1])
    assert math.isnan(value["nested"]["r2"]) and isinstance(value["tuple"], tuple)


@pytest.mark.parametrize("module", ["custom_model_api", "quick_tools_api", "advanced_tools_api"])
def test_model_routes_share_strict_event_encoding(module):
    from importlib import import_module
    api = import_module("backend.web_v2." + module)
    payload = {"metric": float("nan"), "message": "测试\n第二行", "done": False}
    assert decode(api._sse("metrics", payload)) == {"metric": None, "message": "测试\n第二行", "done": False}
