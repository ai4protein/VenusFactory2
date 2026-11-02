"""Explicit version dispatch for measured-context models; no inferred upgrades."""

PLUGINS = frozenset({"generated_context.v1", "generated_context.v2", "generated_context.v3"})


def module_for(plugin_id):
    if plugin_id == "generated_context.v1":
        try:
            from agent.rsi import generated_context as module
        except ImportError:  # Standalone capability bundle.
            import generated_context as module
    elif plugin_id == "generated_context.v2":
        try:
            from agent.rsi import generated_context_v2 as module
        except ImportError:
            import generated_context_v2 as module
    elif plugin_id == "generated_context.v3":
        try:
            from agent.rsi import generated_context_v3 as module
        except ImportError:
            import generated_context_v3 as module
    elif plugin_id == "generated_assay.v1":
        try:
            from agent.rsi import generated_assay as module
        except ImportError:
            import generated_assay as module
    elif plugin_id == "generated_antibody.v1":
        try:
            from agent.rsi import generated_antibody as module
        except ImportError:
            import generated_antibody as module
    else:
        raise ValueError("Unsupported contextual model interface")
    return module


def scope_plugin(scope):
    if not isinstance(scope, dict):
        raise ValueError("Context scope must be a mapping")
    if "interface_version" not in scope:
        return "generated_context.v1"
    if scope["interface_version"] not in {
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        raise ValueError("Unsupported explicit context scope version")
    return scope["interface_version"]


def validate_scope(scope):
    return module_for(scope_plugin(scope)).validate_scope(scope)


def validate_rows(rows, scope, *, labels):
    return module_for(scope_plugin(scope)).validate_rows(rows, scope, labels=labels)
