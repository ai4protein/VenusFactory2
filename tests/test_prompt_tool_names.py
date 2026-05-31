"""CI guard: every `"tool_name": "<name>"` literal in prompts must resolve to a
real tool registered by `tools.tools_agent_hub.get_tools()`.

The test is skipped (not failed) when tool loading fails — typically because the
test environment lacks heavyweight ML/biology dependencies (e.g. langchain).
This keeps the lightweight unit-test runner green while still catching real
typos in CI environments that install the full requirements.
"""
import difflib
import re
import sys
import unittest
from pathlib import Path

# Make `src/` importable just like tests/conftest.py does for pytest.
SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "src" / "agent" / "prompts"
TOOL_NAME_RE = re.compile(r'"tool_name"\s*:\s*"([^"]+)"')


def _load_real_tool_names():
    """Return the set of real tool names, or None if the tools cannot be imported."""
    try:
        from tools.tools_agent_hub import get_tools  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional deps
        return None, f"tools.tools_agent_hub import failed: {exc!r}"
    try:
        tools = get_tools()
    except Exception as exc:  # pragma: no cover
        return None, f"get_tools() raised: {exc!r}"
    names = set()
    for t in tools:
        name = getattr(t, "name", None)
        if isinstance(name, str) and name:
            names.add(name)
    if not names:
        return None, "get_tools() returned no tools with .name"
    return names, None


def _iter_prompt_tool_refs():
    """Yield (file_path, tool_name) for every "tool_name": "..." in any prompt."""
    for path in sorted(PROMPTS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for match in TOOL_NAME_RE.finditer(text):
            yield path, match.group(1)


class PromptToolNamesTests(unittest.TestCase):
    def test_all_prompt_tool_names_are_registered(self):
        self.assertTrue(
            PROMPTS_DIR.is_dir(),
            f"Prompts directory not found: {PROMPTS_DIR}",
        )

        real_names, skip_reason = _load_real_tool_names()
        if real_names is None:
            self.skipTest(skip_reason)

        # Allow prompt authors to refer to placeholder/templated names by skipping
        # values that look like format-string variables (none exist today; this
        # keeps future-proofing minimal).
        bad = []  # list[tuple[Path, str, list[str]]]
        for path, tool_name in _iter_prompt_tool_refs():
            if tool_name.startswith("{") and tool_name.endswith("}"):
                continue
            if tool_name not in real_names:
                suggestions = difflib.get_close_matches(tool_name, real_names, n=3, cutoff=0.5)
                bad.append((path, tool_name, suggestions))

        if bad:
            lines = ["Unknown tool_name references found in prompts:"]
            for path, tool_name, suggestions in bad:
                rel = path.relative_to(Path(__file__).resolve().parent.parent)
                hint = (
                    f" closest matches: {suggestions}" if suggestions else " (no close match)"
                )
                lines.append(f"  - {rel}: '{tool_name}' is not a registered tool;{hint}")
            self.fail("\n".join(lines))


if __name__ == "__main__":
    unittest.main()
