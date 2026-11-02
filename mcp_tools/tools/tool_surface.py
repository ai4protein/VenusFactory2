"""Public tool identities and compatibility names, independent of a harness."""

RETIRED_CODE_TOOLS = frozenset({'agent_generated_code', 'python_repl', 'Python_REPL'})

# Old research receipts retain these names; new MCP catalogs expose one interface.
WORKSPACE_ALIASES = {
    'list_research_files': 'list_session_files',
    'read_research_file': 'read_session_file',
    'write_research_file': 'write_session_file',
    'run_research_python': 'run_session_python',
}

CODE_MIGRATION_MESSAGE = (
    'Write the Python source with write_session_file(path, content), then call '
    'run_session_python(path, timeout_seconds). The server assigns the session workspace.'
)
