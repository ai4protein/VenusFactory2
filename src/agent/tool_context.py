"""
Tool execution context with generic dependency injection.

Provides a ``ToolContext[TContext]`` that threads user-defined state
through the tool execution lifecycle without the tool framework needing
to know the concrete type. Inspired by OpenAI Agents SDK's
``RunContextWrapper``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

TContext = TypeVar("TContext")


@dataclass
class UsageStats:
    """Accumulated token / cost usage across a run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    tool_calls: int = 0
    llm_calls: int = 0

    def add_llm_usage(
        self, prompt: int = 0, completion: int = 0, total: int = 0
    ) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += total or (prompt + completion)
        self.llm_calls += 1

    def add_tool_call(self) -> None:
        self.tool_calls += 1


@dataclass
class ToolContext(Generic[TContext]):
    """Context object threaded through tool execution.

    Attributes:
        context: User-supplied mutable state (any type).
        session_id: Current session identifier.
        session_dir: Filesystem path for session artifacts.
        usage: Accumulated usage statistics.
        step_results: Results from prior pipeline steps (keyed by step index).
        tool_cache: Shared cache for deduplicating repeated tool calls.
        metadata: Arbitrary key-value store for ad-hoc data.
    """

    context: TContext
    session_id: str = ""
    session_dir: str = ""
    usage: UsageStats = field(default_factory=UsageStats)
    step_results: dict[int, Any] = field(default_factory=dict)
    tool_cache: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def session_path(self) -> Path:
        return Path(self.session_dir)

    def get_step_output(self, step_index: int) -> Any | None:
        result = self.step_results.get(step_index)
        if result is None:
            return None
        if isinstance(result, dict):
            return result.get("raw_output")
        return result

    def set_step_output(self, step_index: int, output: Any) -> None:
        self.step_results[step_index] = {"raw_output": output}

    def get_cached(self, key: str) -> Any | None:
        return self.tool_cache.get(key)

    def set_cached(self, key: str, value: Any) -> None:
        self.tool_cache[key] = value


@dataclass
class ProteinToolContext(ToolContext[Any]):
    """Specialized context for protein engineering tools.

    Carries protein-specific state that tools commonly need.
    """

    sequence: str = ""
    pdb_id: str = ""
    uniprot_id: str = ""
    organism: str = ""
    ui_lang: str = "en"

    def update_from_protein_context(self, protein_ctx: Any) -> None:
        """Sync fields from the legacy ProteinContextManager."""
        if protein_ctx is None:
            return
        ctx = protein_ctx
        if hasattr(ctx, "context") and isinstance(ctx.context, dict):
            ctx = ctx.context
        if isinstance(ctx, dict):
            self.sequence = ctx.get("sequence", self.sequence) or self.sequence
            self.pdb_id = ctx.get("pdb_id", self.pdb_id) or self.pdb_id
            self.uniprot_id = ctx.get("uniprot_id", self.uniprot_id) or self.uniprot_id
            self.organism = ctx.get("organism", self.organism) or self.organism
