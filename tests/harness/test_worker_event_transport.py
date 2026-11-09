"""Worker events remain readable when diagnostics or an adapter fail."""
import builtins
import contextlib
import io
import json
import sys

import pytest

from agent.harness import events, web_worker


class Transport(io.StringIO):
    def __init__(self):
        super().__init__()
        self.flushes = 0

    def flush(self):
        self.flushes += 1
        return super().flush()


def test_worker_event_is_one_flushed_json_record_not_a_diagnostic(monkeypatch):
    transport, diagnostics = Transport(), io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", transport)
    value = {"kind": "text", "text": "蛋白质\nSecond line"}
    with contextlib.redirect_stdout(diagnostics):
        print("tool diagnostic")
        events.emit(value)
    assert diagnostics.getvalue() == "tool diagnostic\n"
    assert len(transport.getvalue().splitlines()) == 1
    assert json.loads(transport.getvalue()) == value
    assert transport.flushes == 1


def test_entrypoint_reports_error_without_importing_another_adapter(monkeypatch):
    transport, diagnostics = Transport(), io.StringIO()
    monkeypatch.setattr(sys, "__stdout__", transport)
    monkeypatch.setattr(sys, "stderr", diagnostics)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"python_paths": list(sys.path)})))
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.endswith(("web_langgraph_worker", "web_native_worker")):
            raise AssertionError("error transport must not import an adapter")
        return real_import(name, *args, **kwargs)

    async def failing_run(request):
        print("worker diagnostic")
        raise ValueError("private exception details")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(web_worker, "run", failing_run)
    with pytest.raises(SystemExit) as error:
        web_worker.main()
    assert error.value.code == 1
    assert json.loads(transport.getvalue()) == {"kind": "error", "error_type": "ValueError"}
    assert diagnostics.getvalue() == "worker diagnostic\n"
