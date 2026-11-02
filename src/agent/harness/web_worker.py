"""Web subprocess entry: preserve the host's runtime paths and cancellation."""

import asyncio
import contextlib
import json
import signal
import sys


async def run(request):
    task = asyncio.current_task()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    if request["profile"]["harness"] == "langgraph":
        from agent.harness.web_langgraph_worker import run as execute
    else:
        from agent.harness.web_native_worker import run as execute
    await execute(request)


def main():
    request = json.loads(sys.stdin.readline())
    sys.path[:] = request.pop("python_paths")
    from agent.harness.web_langgraph_worker import emit

    with contextlib.redirect_stdout(sys.stderr):
        try:
            asyncio.run(run(request))
        except (Exception, asyncio.CancelledError) as exc:
            emit({"kind": "error", "error_type": type(exc).__name__})
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
