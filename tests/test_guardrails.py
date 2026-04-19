"""Tests for the guardrails framework."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from agent.guardrails import (
    GuardrailResult,
    InputGuardrail,
    OutputGuardrail,
    code_safety_guardrail,
    input_guardrail,
    input_size_guardrail,
    output_guardrail,
    path_traversal_guardrail,
    run_input_guardrails,
    run_output_guardrails,
    DEFAULT_INPUT_GUARDRAILS,
)
from exceptions import InputGuardrailTripped, OutputGuardrailTripped


class TestGuardrailResult:
    def test_passed(self):
        r = GuardrailResult.passed("ok")
        assert not r.tripwire_triggered
        assert r.message == "ok"

    def test_tripped(self):
        r = GuardrailResult.tripped("bad input", output_info={"key": "val"})
        assert r.tripwire_triggered
        assert r.output_info == {"key": "val"}


class TestDecorators:
    def test_input_guardrail_bare(self):
        @input_guardrail
        def my_guard(tool_name, tool_input, **kwargs):
            return GuardrailResult.passed()

        assert isinstance(my_guard, InputGuardrail)
        assert my_guard.name == "my_guard"

    def test_input_guardrail_with_args(self):
        @input_guardrail(name="custom", run_in_parallel=False)
        def my_guard(tool_name, tool_input, **kwargs):
            return GuardrailResult.passed()

        assert my_guard.name == "custom"
        assert not my_guard.run_in_parallel

    def test_output_guardrail_bare(self):
        @output_guardrail
        def my_guard(tool_name, tool_output, **kwargs):
            return GuardrailResult.passed()

        assert isinstance(my_guard, OutputGuardrail)


class TestCodeSafetyGuardrail:
    @pytest.mark.asyncio
    async def test_safe_code_passes(self):
        result = await code_safety_guardrail.run(
            "python_repl", {"query": "import numpy as np\nprint(np.array([1,2,3]))"}
        )
        assert not result.tripwire_triggered

    @pytest.mark.asyncio
    async def test_os_system_blocked(self):
        result = await code_safety_guardrail.run(
            "python_repl", {"query": "os.system('rm -rf /')"}
        )
        assert result.tripwire_triggered

    @pytest.mark.asyncio
    async def test_subprocess_blocked(self):
        result = await code_safety_guardrail.run(
            "agent_generated_code", {"code": "subprocess.run(['ls'])"}
        )
        assert result.tripwire_triggered

    @pytest.mark.asyncio
    async def test_eval_blocked(self):
        result = await code_safety_guardrail.run(
            "python_repl", {"query": "eval('__import__(\"os\").system(\"ls\")')"}
        )
        assert result.tripwire_triggered

    @pytest.mark.asyncio
    async def test_non_code_tool_passes(self):
        result = await code_safety_guardrail.run(
            "blast_search", {"query": "os.system('rm')"}
        )
        assert not result.tripwire_triggered


class TestPathTraversalGuardrail:
    @pytest.mark.asyncio
    async def test_normal_path_passes(self):
        result = await path_traversal_guardrail.run(
            "read_fasta", {"file_path": "/home/user/data/protein.fasta"}
        )
        assert not result.tripwire_triggered

    @pytest.mark.asyncio
    async def test_traversal_blocked(self):
        result = await path_traversal_guardrail.run(
            "read_fasta", {"file_path": "/home/user/../../etc/passwd"}
        )
        assert result.tripwire_triggered

    @pytest.mark.asyncio
    async def test_sensitive_dir_blocked(self):
        result = await path_traversal_guardrail.run(
            "read_fasta", {"file_path": "/etc/shadow"}
        )
        assert result.tripwire_triggered

    @pytest.mark.asyncio
    async def test_non_path_param_ignored(self):
        result = await path_traversal_guardrail.run(
            "blast_search", {"sequence": "/etc/passwd"}
        )
        assert not result.tripwire_triggered


class TestInputSizeGuardrail:
    @pytest.mark.asyncio
    async def test_normal_input_passes(self):
        result = await input_size_guardrail.run(
            "blast", {"sequence": "ACDEFGHIK" * 100}
        )
        assert not result.tripwire_triggered

    @pytest.mark.asyncio
    async def test_huge_input_blocked(self):
        result = await input_size_guardrail.run(
            "blast", {"sequence": "A" * 20_000_000}, max_input_bytes=10_000_000
        )
        assert result.tripwire_triggered


class TestRunGuardrails:
    @pytest.mark.asyncio
    async def test_all_pass(self):
        results = await run_input_guardrails(
            DEFAULT_INPUT_GUARDRAILS,
            "blast_search",
            {"sequence": "ACDEFGHIK"},
        )
        assert all(not r.tripwire_triggered for r in results)

    @pytest.mark.asyncio
    async def test_trips_on_violation(self):
        with pytest.raises(InputGuardrailTripped):
            await run_input_guardrails(
                DEFAULT_INPUT_GUARDRAILS,
                "python_repl",
                {"query": "os.system('rm -rf /')"},
            )

    @pytest.mark.asyncio
    async def test_output_guardrails(self):
        @output_guardrail
        def check_empty(tool_name, tool_output, **kwargs):
            if not tool_output:
                return GuardrailResult.tripped("Empty output")
            return GuardrailResult.passed()

        with pytest.raises(OutputGuardrailTripped):
            await run_output_guardrails([check_empty], "blast", None)

    @pytest.mark.asyncio
    async def test_async_guardrail(self):
        @input_guardrail
        async def async_guard(tool_name, tool_input, **kwargs):
            return GuardrailResult.passed("async ok")

        results = await run_input_guardrails([async_guard], "test", {})
        assert len(results) == 1
        assert results[0].message == "async ok"
