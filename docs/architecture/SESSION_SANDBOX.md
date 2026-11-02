# Session code lifecycle and host protection

Session-authored code is invoked only through `write_session_file` followed by
`run_session_python`, or the bounded `run_session_shell` tool. It does not run in
the API process or through an unrestricted harness shell.

## Lifetime

1. Validate the active task and the session-owned workspace.
2. Wait for the existing CPU job slot and host-wide sandbox admission.
3. Create a fresh bwrap namespace only when execution can start.
4. Mount only that session at `/workspace`, a private scratch directory at
   `/tmp`, and read-only scientific libraries and host-owned runner code.
5. On success, failure, timeout or cancellation, kill/reap the execution, remove
   scratch files and release the reservation. Keep session result files and
   bounded logs/receipts outside the guest mount.

SSE connections subscribe to persisted progress; they do not own compute slots.
Refreshing or switching pages may leave an authorized background task running.
An idle, stopped, paused, completed or input-waiting task must not start a new
sandbox. Running code checks task cancellation every 250 ms. The worker queue
also stops descendants when a parent job is cancelled or its lease is lost.
There is no preallocated code sandbox per conversation or SSE connection.

## Shared admission

`runtime/control/sandboxes/` is private host scheduling state shared by local
and online instances of the same checkout. User files remain separated under
their mode/session roots. Multiple checkouts on the same host must set the same
`VENUS_SANDBOX_CONTROL_ROOT` to share this budget.

Defaults are two simultaneous code sandboxes, one per owner/workspace, and
16 GiB of total reserved address space. Online code receives 2 GiB; local code
receives 8 GiB. Admission conservatively accounts for existing reservations,
host available memory **and the worker's cgroup memory headroom**, leaving
4 GiB free. It also requires at least 2 GiB free on the relevant filesystems.
Waiting is bounded to 60 seconds and creates no guest process. Explicitly
stopped queued work cannot subsequently start a guest.

Reservations use a cross-process file lock and Linux boot/process start
identities, not a process-local semaphore or PID alone. Dead worker records are
reclaimed on the next admission check. Per-run receipts include the outcome,
elapsed time, configured memory and temporary-directory cleanup result.

## Enforced execution profile

- Namespaces isolate the network, process view and mounts; credentials and host
  configuration are not passed to the supervisor or guest.
- Root, `/proc`, `/dev`, runtime libraries and runner code are read-only.
  `/tmp` is disk-backed scratch, not an unbounded guest-writable tmpfs.
- Kernel resource limits bound address space, CPU time, file size and open
  descriptors. Each execution is pinned to one allowed CPU; BLAS threads default
  to one. Online wall time is at most 60 seconds; local requests at most 300.
- A mandatory seccomp filter prevents forks, process-like clone, nested
  namespaces, mount changes, ptrace/process-memory operations, memfd and SysV
  shared-memory creation, kernel modules, BPF/perf and io_uring. Threads sharing
  the same address space remain possible. Missing seccomp support stops code
  execution; it never falls back to a host shell.
- This is a single-process analysis profile. Use `n_jobs=1`, direct Python and
  small increments. Background jobs, shell pipelines and multiprocessing are
  not available. Installed models and multiprocess training use their separate
  administrator-enabled, queued MCP tools.
- Output is capped at 2 MiB. Workspace/scratch storage and free memory/disk are
  monitored during execution. Storage checks are periodic safeguards, not a
  filesystem quota. Upload limits, per-IP daily conversations and task budgets
  remain independent controls.

## Security boundary and remaining deployment work

Prompt instructions, web pages and uploaded documents cannot grant filesystem
authority, choose another session mount or enable disabled tools. Authorization,
path validation, tool allowlists and sandbox restrictions are host-enforced.
The tests use synthetic credentials, cross-session files, forbidden syscall
attempts, bounded memory requests and cancellation races; no destructive fork
bomb or host memory-exhaustion test is necessary.

Markdown is sanitized before rendering. Untrusted HTML/SVG artifact responses
also carry a server-enforced sandbox CSP, no-sniff and no-referrer headers; this
applies to direct file URLs as well as in-page previews. The policy disables
scripts, forms, same-origin privileges and outbound fetches. User artifacts do
not receive the platform UI's JavaScript permissions.

Bubblewrap shares the host kernel. It is not a VM and does not by itself provide
aggregate resource control or protect against every kernel vulnerability. The
current hl_gpu account cannot create delegated cgroups; the strict single-process
profile prevents fork-based multiplication of its per-process address-space
limit. Do not describe this as a dedicated cgroup or VM boundary.

For anonymous production deployment, use an isolated compute host/account,
maintained kernel/bubblewrap packages, delegated cgroup CPU/memory/pids quotas,
filesystem quotas and ingress rate/concurrency limits. Broader subprocess support
must be implemented with those quotas before relaxing the syscall policy.

References: [Bubblewrap security model](https://github.com/containers/bubblewrap#sandbox-security),
[Linux resource limits](https://man7.org/linux/man-pages/man2/setrlimit.2.html),
[seccomp filters](https://www.kernel.org/doc/html/latest/userspace-api/seccomp_filter.html).
