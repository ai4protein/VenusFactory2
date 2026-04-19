"""Tests for tool execution context and dependency injection."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from agent.tool_context import (
    UsageStats,
    ToolContext,
    ProteinToolContext,
)


class TestUsageStats:
    def test_defaults(self):
        u = UsageStats()
        assert u.prompt_tokens == 0
        assert u.tool_calls == 0
        assert u.llm_calls == 0

    def test_add_llm_usage(self):
        u = UsageStats()
        u.add_llm_usage(prompt=100, completion=50)
        assert u.prompt_tokens == 100
        assert u.completion_tokens == 50
        assert u.total_tokens == 150
        assert u.llm_calls == 1

    def test_add_llm_usage_with_total(self):
        u = UsageStats()
        u.add_llm_usage(prompt=100, completion=50, total=200)
        assert u.total_tokens == 200

    def test_add_tool_call(self):
        u = UsageStats()
        u.add_tool_call()
        u.add_tool_call()
        assert u.tool_calls == 2

    def test_accumulates(self):
        u = UsageStats()
        u.add_llm_usage(prompt=10, completion=5)
        u.add_llm_usage(prompt=20, completion=10)
        assert u.prompt_tokens == 30
        assert u.completion_tokens == 15
        assert u.llm_calls == 2


class TestToolContext:
    def test_basic_creation(self):
        ctx = ToolContext(context={"key": "value"}, session_id="s1")
        assert ctx.context == {"key": "value"}
        assert ctx.session_id == "s1"

    def test_generic_context_type(self):
        class MyState:
            value: int = 42

        state = MyState()
        ctx = ToolContext[MyState](context=state)
        assert ctx.context.value == 42

    def test_step_results(self):
        ctx = ToolContext(context=None)
        ctx.set_step_output(0, {"file": "out.csv"})
        assert ctx.get_step_output(0) == {"file": "out.csv"}
        assert ctx.get_step_output(99) is None

    def test_caching(self):
        ctx = ToolContext(context=None)
        assert ctx.get_cached("key") is None
        ctx.set_cached("key", [1, 2, 3])
        assert ctx.get_cached("key") == [1, 2, 3]

    def test_session_path(self):
        ctx = ToolContext(context=None, session_dir="/tmp/sess")
        assert str(ctx.session_path) == "/tmp/sess"


class TestProteinToolContext:
    def test_defaults(self):
        ctx = ProteinToolContext(context=None)
        assert ctx.sequence == ""
        assert ctx.pdb_id == ""
        assert ctx.ui_lang == "en"

    def test_with_protein_data(self):
        ctx = ProteinToolContext(
            context=None,
            sequence="ACDEFGHIK",
            pdb_id="1ABC",
            uniprot_id="P12345",
        )
        assert ctx.sequence == "ACDEFGHIK"
        assert ctx.pdb_id == "1ABC"

    def test_update_from_dict_context(self):
        ctx = ProteinToolContext(context=None)
        mock_protein_ctx = {
            "sequence": "MKTL",
            "pdb_id": "2XYZ",
            "uniprot_id": "Q99999",
            "organism": "E. coli",
        }
        ctx.update_from_protein_context(mock_protein_ctx)
        assert ctx.sequence == "MKTL"
        assert ctx.pdb_id == "2XYZ"
        assert ctx.uniprot_id == "Q99999"
        assert ctx.organism == "E. coli"

    def test_update_from_none(self):
        ctx = ProteinToolContext(context=None, sequence="original")
        ctx.update_from_protein_context(None)
        assert ctx.sequence == "original"

    def test_inherits_tool_context(self):
        ctx = ProteinToolContext(context="state", session_id="s1")
        ctx.set_cached("k", "v")
        assert ctx.get_cached("k") == "v"
        assert isinstance(ctx, ToolContext)
