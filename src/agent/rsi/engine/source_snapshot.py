"""Complete source closure for isolated scientific execution snapshots."""

from pathlib import Path


def research_implementation_paths(root):
    """Include role prompt assets alongside imported code and research entry points.

    Snapshot source and the public free-route policy: model credentials, deployment configuration,
    historical results and scientific datasets are admitted separately.
    """
    root = Path(root).resolve()
    return sorted(
        {
            *list((root / "scripts/rsi").rglob("*.py")),
            *list((root / "configs/rsi").glob("model_access.json")),
            *list((root / "configs/harness").glob("free_execution_catalog.json")),
            *list((root / "configs/rsi").glob("scientist_tool_policy_*.json")),
            *(path for area in ("src", "backend", "mcp_tools")
              for path in (root / area).rglob("*.py")
              if "vendor" not in path.parts and "__pycache__" not in path.parts),
            *list((root / "src").glob("constant.json")),
            *list((root / "src/agent/prompts").glob("*.md")),
        }
    )
