"""Versioned feedback routing without changing legacy numerical replay."""

import hashlib
import json
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

try:
    from agent.rsi import context_feedback_profiles as legacy
except ImportError:  # Standalone capability bundle.
    import context_feedback_profiles as legacy

FIELD = legacy.FIELD
POLICY = legacy.POLICY
EXPRESSION_POLICY = legacy.EXPRESSION_POLICY
WIDE_POLICY = "group_baselines.v2"
CONDITION_POLICY = "group_baselines.v3"
POLICIES = (*legacy.POLICIES, WIDE_POLICY, CONDITION_POLICY)


def wide_module():
    try:
        from agent.rsi import context_v2_feedback as module
    except ImportError:
        import context_v2_feedback as module
    return module


def condition_module():
    try:
        from agent.rsi import context_v3_feedback as module
    except ImportError:
        import context_v3_feedback as module
    return module


def selected(declaration, plugin_id=None):
    plugin = plugin_id or declaration.get("model_plugin_id", declaration.get("plugin_id"))
    if plugin == "generated_context.v3":
        return condition_module()
    return wide_module() if plugin == "generated_context.v2" else legacy


def policy_for(declaration, *, plugin_id=None):
    return selected(declaration, plugin_id).policy_for(declaration, plugin_id=plugin_id)


def verify_goal(state, protocol):
    return selected(protocol).verify_goal(state, protocol)


def verify_bundle(task, protocol):
    return selected(protocol).verify_bundle(task, protocol)


def description(scope, *, policy=POLICY):
    if policy == CONDITION_POLICY:
        return condition_module().description(scope)
    if policy == WIDE_POLICY:
        return wide_module().description(scope)
    return legacy.description(scope, policy=policy)


def comparison(rows, scope, candidate_metrics, *, policy=POLICY):
    if policy == CONDITION_POLICY:
        return condition_module().comparison(rows, scope, candidate_metrics)
    if policy == WIDE_POLICY:
        return wide_module().comparison(rows, scope, candidate_metrics)
    return legacy.comparison(rows, scope, candidate_metrics, policy=policy)


# A numerical replay is reusable only during one publication/resolution audit.
# Retain identities and source bindings, never model rows or a cross-request verdict.
_MAX_VERIFIED_FEEDBACK = 256
_VERIFIED_FEEDBACK = ContextVar("context_feedback_verification", default=None)


@contextmanager
def verification_session():
    if _VERIFIED_FEEDBACK.get() is not None:
        yield
        return
    token = _VERIFIED_FEEDBACK.set(OrderedDict())
    try:
        yield
    finally:
        _VERIFIED_FEEDBACK.reset(token)


def _sources_match(implementation):
    if not isinstance(implementation, dict) or not 1 <= len(implementation) <= 8:
        return False
    root = Path(__file__).parent
    for name, expected in implementation.items():
        if not isinstance(name, str) or Path(name).name != name:
            return False
        path = root / name
        try:
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return False
        except OSError:
            return False
    return True


def _feedback_key(task, rows, feedback):
    values = [task, rows, feedback]
    try:
        payload = json.dumps(values, sort_keys=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    # JSON encoding erases tuple/list and numeric-subclass distinctions. Such
    # Python inputs must still pass the original verifier on every invocation.
    pending = [values]
    while pending:
        value = pending.pop()
        if type(value) is dict:
            if any(type(name) is not str for name in value):
                return None
            pending.extend(value.values())
        elif type(value) is list:
            pending.extend(value)
        elif type(value) not in (str, int, float, bool, type(None)):
            return None
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_feedback(task, rows, feedback):
    module = selected(task)
    verified = _VERIFIED_FEEDBACK.get()
    if verified is None:
        return module.verify_feedback(task, rows, feedback)
    key = _feedback_key(task, rows, feedback)
    if key is None:
        return module.verify_feedback(task, rows, feedback)
    if key in verified and _sources_match(verified[key]):
        verified.move_to_end(key)
        return None

    # Only the unchanged versioned verifier can establish a successful result.
    result = module.verify_feedback(task, rows, feedback)
    packet = feedback.get("baseline_comparison")
    implementation = packet.get("implementation") if isinstance(packet, dict) else None
    if _sources_match(implementation):
        verified[key] = dict(implementation)
        verified.move_to_end(key)
        while len(verified) > _MAX_VERIFIED_FEEDBACK:
            verified.popitem(last=False)
    return result
