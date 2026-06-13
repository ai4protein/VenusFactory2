"""Modular graph package.

Subpackages such as ``execution`` host implementations extracted from the
historical monolithic ``agent.chat_graph`` module. Other node families
(plan/research/chat/finalize) live alongside as ``research``/``chat``/
``planning``/``finalize`` modules.

Module-level state kept on the package root (single shared instance):
- ``_BG_TASKS``: strong references for fire-and-forget hook tasks
- ``_TOOL_TIMEOUTS``: per-tool soft timeout overrides
"""
from __future__ import annotations

import asyncio

# Strong references for fire-and-forget background tasks (hook dispatch).
# Without this, the asyncio.create_task tasks can be garbage-collected
# mid-flight.
_BG_TASKS: set[asyncio.Task] = set()

_TOOL_TIMEOUTS: dict[str, int] = {
    "query_literature_by_keywords": 60,
    "query_pubmed": 60,
    "query_semantic_scholar": 60,
    "query_arxiv": 60,
    "query_tavily": 30,
    "query_duckduckgo": 30,
    "query_github": 30,
    "query_hugging_face": 30,
    "query_fda_by_keywords": 30,
    "query_web_by_keywords": 30,
    "query_biorxiv": 60,
    "download_brenda_km_values_by_ec_number": 120,
    "download_brenda_specific_activity_by_ec_number": 120,
    "query_foldseek_search_by_pdb_file": 300,
    "query_sequence_from_pdb_file": 120,
    "esmfold_structure_prediction": 300,
    "zero_shot_mutation_sequence_prediction": 600,
    # ProSST structure-based scan is O(L) on residues — 300s is too tight for
    # proteins like EGFR (1210 aa). Bump to 1200s. Per-step retry still applies.
    "zero_shot_mutation_structure_prediction": 1200,
    "train_protein_model": 600,
    "protein_model_predict": 300,
    "agent_generated_code": 300,
}


def __getattr__(name: str):  # pragma: no cover - lazy convenience
    """Allow `from agent.graph import create_agent_graph` lazily.

    Importing :mod:`agent.graph.compile` eagerly would force every importer of
    this package to also pull LangGraph, langchain, and every node module —
    expensive when callers only want shared state (``_BG_TASKS``).
    """
    if name == "create_agent_graph":
        from agent.graph.compile import create_agent_graph

        return create_agent_graph
    if name == "AgentState":
        from agent.graph.state import AgentState

        return AgentState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["_BG_TASKS", "_TOOL_TIMEOUTS", "create_agent_graph", "AgentState"]
