"""Bounded child process groups with cancellation propagated by CLI entry points."""

import os
import signal
import subprocess

MPI_TERMINATION_GRACE = 0.5
STAGE_TERMINATION_GRACE = 2.0


class TerminationRequested(RuntimeError):
    pass


def install_termination_handlers():
    def interrupted(signum, frame):
        raise TerminationRequested(f"Controller received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)


def stop_group(process, grace):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The group may retain descendants after its leader has exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_in_group(argv, *, cwd, env, log, stdin=None, timeout=180, termination_grace=STAGE_TERMINATION_GRACE):
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        text=True,
        start_new_session=True,
    )
    try:
        process.communicate(stdin, timeout=timeout)
        return process.returncode
    finally:
        stop_group(process, termination_grace)
