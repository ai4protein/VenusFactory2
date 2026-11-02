"""Versioned feedback routing without changing legacy numerical replay."""

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


def verify_feedback(task, rows, feedback):
    return selected(task).verify_feedback(task, rows, feedback)
