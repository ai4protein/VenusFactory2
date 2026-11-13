"""Side-effect-free admission for operator-configured local cluster execution."""

from __future__ import annotations

import os
from collections.abc import Mapping


def require_local_execution(environment: Mapping[str, str] | None = None) -> None:
    """Reject online/unknown modes before reading configuration or cluster state.

    An unset deployment mode preserves the standalone local CLI default. This is
    an entry-point check, not a cleanup gate for a previously admitted local job.
    """
    environment = os.environ if environment is None else environment
    mode = environment.get("WEBUI_V2_MODE", "local")
    online = environment.get("VENUS_SCIENTIST_ONLINE", "0")
    if (
        not isinstance(mode, str)
        or mode.strip().casefold() != "local"
        or not isinstance(online, str)
        or online.strip().casefold() not in {"", "0", "false", "no", "off"}
    ):
        raise PermissionError("Cluster execution is available only in local deployment mode")
