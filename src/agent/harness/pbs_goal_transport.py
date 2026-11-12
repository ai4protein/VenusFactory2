"""Bounded, noninteractive PBS command transport for POSIX hosts."""
from __future__ import annotations

import locale
import math
import os
import selectors
import signal
import subprocess
import time

MAX_STDOUT_BYTES = 16 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
_READ_BYTES = 64 * 1024


def _stop_group(process):
    # Keep the session leader unreaped until after signalling its group. A child
    # may still own a pipe after the leader exits; the leader PID cannot be reused.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def bounded_run(argv, *, timeout, env, capture_output=True, text=True):
    """Capture both streams within one deadline; never attach output to errors."""
    if capture_output is not True or type(text) is not bool:
        raise ValueError("PBS commands require captured output and an explicit text mode")
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("PBS command timeout must be positive and finite")
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, start_new_session=True,
        shell=False,
    )
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": MAX_STDOUT_BYTES, "stderr": MAX_STDERR_BYTES}
    selector = None
    try:
        selector = selectors.DefaultSelector()
        for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout)
            ready = selector.select(remaining)
            for key, _ in ready:
                try:
                    chunk = os.read(key.fd, _READ_BYTES)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                name = key.data
                if len(buffers[name]) + len(chunk) > limits[name]:
                    raise RuntimeError("PBS command output exceeded its byte limit")
                buffers[name].extend(chunk)
        process.wait(timeout=max(0, deadline - time.monotonic()))
    except BaseException:
        _stop_group(process)
        raise
    finally:
        if selector is not None:
            selector.close()
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()
    output = [bytes(buffers[name]) for name in ("stdout", "stderr")]
    if text:
        encoding = locale.getpreferredencoding(False)
        output = [value.decode(encoding, errors="replace").replace("\r\n", "\n").replace("\r", "\n")
                  for value in output]
    return subprocess.CompletedProcess(argv, process.returncode, *output)
