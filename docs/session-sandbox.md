# Session execution isolation

Web Agent and Scientist use session-scoped file tools and the same isolated
Python execution service. Agent also has a bounded shell tool. The harness owns
the dialogue, but does not receive a generic host Python REPL or shell. Native
Kimi, DeepSeek, Codex and Claude adapters advertise only admitted MCP functions.
Old Web execution entry points cannot resume an unsandboxed daemon; a follow-up
uses the isolated adapter and the existing conversation transcript.

## MCP session contract

Sandbox execution is exposed through the authenticated per-session MCP gateway:

`Harness -> MCP tools/call -> CPU queue -> local sandbox -> session artifacts`

Agent and Scientist share `list_session_files`, `read_session_file`,
`write_session_file`, `edit_session_file` and `run_session_python`. Agent also
supports the bounded `run_session_shell` tool. Execution receipts support review.
Both execution tools use the same sandbox service.

For a small correction, read the file and call `edit_session_file` with its
full-file `sha256` (or `file_sha256` on a paged preview), the exact old text and
its replacement. The match must be unique and the file must still have that
version. Edits preserve other contents and do not execute code. They support
UTF-8 text files up to 200,000 bytes; use the full hash, never `slice_sha256`.
Conditional edits and ordinary file writes share a directory lock across worker
processes. Stale edits are rejected instead of overwriting a newer tool write.

The backend binds the current conversation/research workspace when creating the
MCP tool instances. Each execution mounts that directory at `/workspace` and
starts there; uploads and generated files persist for subsequent calls. The
caller supplies only the declared file, code/command and execution options.
There is no caller-selectable host directory, mount list, session ID, credential
or sandbox backend. Unexpected fields are rejected instead of silently ignored.
Each MCP endpoint has its own credential; another session's credential cannot
authorize a call. The public shared MCP catalog does not expose a generic
host-path sandbox launcher.

The trusted Web MCP broker validates input and output paths against the current
session before invoking scientific services. Traversal, hidden paths and symlink
escapes are rejected. Tool results use `/workspace` paths instead of host paths.
The broker and compute workers still need their own reviewed implementations:
session isolation does not make an arbitrary host-side plugin trustworthy.

## Running now: Bubblewrap

`VENUS_SESSION_SANDBOX=bwrap` is the default on Linux. The child sees only its
session at `/workspace`, a read-only Python installation, standard binaries,
fresh PID/network namespaces and temporary directories. Host home directories,
repository, other sessions, credentials, GPU devices and the host network are
not mounted into generated-code execution. The controller has no host-process
fallback if Bubblewrap fails. It enforces wall time, per-process CPU/address
space/file-descriptor/process limits, bounded logs, and session file/storage
limits; cancellation kills the process group and namespace descendants.

The launcher itself receives a minimal environment; clearing only the executed
child's environment is insufficient because procfs can expose its supervisor.
The command runs as PID 1 in its private PID namespace. Stdin is `/dev/null`,
stdout/stderr are pipes, and inherited descriptors are closed. The host drains
bounded logs outside the mount. Python binaries and libraries are read-only;
environment metadata and root-level configuration are not mounted.
For virtual environments, the sandbox mounts the base interpreter's standard
library and shared libraries, then overlays the active environment's packages.
The environment roots and pyvenv.cfg remain outside the guest. A system Python
without a site-packages directory uses private, read-only mount points; no
directory is created in the host installation. Native harnesses and MCP
preloading use the same Python runtime layout. The MCP preloader receives a
minimal supervisor environment and selected system certificate/resolver files.

File reads, previews and ZIP downloads use directory-anchored `O_NOFOLLOW`
descriptors, not a validate-then-reopen pathname. Writes use anchored temporary
files and atomic replacement. Symlinks, hard links and special files are refused.
Chat attachments must already belong to the current conversation; an absolute
host path or another conversation's file is not an upload grant. MCP calls that
can use the workspace are serialized until their queued result returns, while
API file downloads remain safe against concurrent pathname replacement.

Regression checks include actual scientific analysis (NumPy, pandas, SciPy,
scikit-learn, Biopython and Matplotlib), persistent CSV/PNG results, procfs
credential canaries, cross-session paths, link substitution, output floods,
cancellation cleanup and authenticated MCP isolation. No LLM is used in these
checks. Run `PYTHONPATH=src VENUS_SESSION_SANDBOX=bwrap python -m pytest -q
tests/harness/test_session_isolation_regressions.py tests/harness/test_session_mcp.py
tests/harness/test_session_boundary.py` on the Linux runtime.

Agent/Scientist files persist between executions. Processes do not. Queued code
jobs share the CPU resource queue. Administrator tool switches include the new
`*_session_*` tools and are checked again when a cached tool is called.

Bubblewrap shares the host kernel. Its per-process limits and storage polling
are not cgroup aggregate quotas or a VM security boundary. Public deployments
of hostile workloads should use a dedicated VM host with cgroup quotas and
filesystem quotas. Do not describe the current deployment as Firecracker.

## Optional: local E2B Embed

Official deployment prerequisites include KVM (`/dev/kvm`), TUN, Docker and
host-level networking setup. Installing the Python SDK alone does not provide
a sandbox. This checkout's current hl_gpu host has no `/dev/kvm` or Docker, so
E2B execution is not enabled or claimed as verified.

On a suitable administrator-managed host, follow:
https://github.com/e2b-dev/runtime/tree/main/embed/compose

Install `requirements/optional/e2b.txt` in the application environment and set:

```
VENUS_SESSION_SANDBOX=e2b
E2B_API_URL=http://127.0.0.1:3000
E2B_SANDBOX_URL=http://127.0.0.1:3002
E2B_API_KEY=<local Embed team key>
VENUS_E2B_TEMPLATE=<reviewed scientific Python template>
```

Use the actual API/proxy URLs emitted by that installation. Both must be local
loopback addresses (a secured tunnel to a dedicated owned VM host also works).
The adapter refuses missing settings and public cloud endpoints. It never sends
files to the default E2B Cloud and never falls back to host execution.

The template must have Python, the required scientific packages, a non-root
`user` and a writable `/workspace`. No LLM credentials belong in the template.
The adapter creates a fresh VM for each execution with outbound and public
inbound traffic denied, transfers only regular session files, validates the
output archive before importing it, and kills the VM on completion/cancellation.
This is a file-transfer workspace, not a direct host-directory bind mount.
Live E2B end-to-end verification remains required after provisioning the host.

## Preferred microVM provider: local CubeSandbox

The requested target is now CubeSandbox. Its native SDK supports an actual
host-directory mount, unlike the file-transfer E2B adapter. Install the optional
client with `pip install -r requirements/optional/cube.txt`; this installs no
hypervisor, kernel, Docker daemon or sandbox service.

After an administrator has deployed CubeAPI, CubeProxy and Cubelet on the same
host as the application, configure the backend only:

```
VENUS_SESSION_SANDBOX=cubesandbox
CUBE_API_URL=http://127.0.0.1:3000
CUBE_PROXY_NODE_IP=127.0.0.1
CUBE_PROXY_PORT_HTTP=80
CUBE_API_KEY=<dedicated local service key>
CUBE_TEMPLATE_ID=<reviewed scientific Python template>
CUBE_COMPUTE_NODE=<actual local Cubelet node ID>
```

The service requires an authenticated, loopback-only control plane. Supplying an
SDK key does not enable authentication on an unconfigured CubeAPI server; its
administrator must enable the server's authentication integration. Do not expose
the installer's default database, scheduler and unauthenticated control ports.

The template must contain Python and the scientific dependencies, with a
non-root `user` whose mapped UID/GID can access the host application's session
directory. The adapter verifies a random mount marker as that user before any
generated code executes. Only the backend-owned session path is placed in the
`host-mount` metadata, at `/workspace`; a single-node scheduling constraint
prevents accidental mounting of the same pathname on an unrelated node.

Every MCP code/shell call creates a fresh VM, denies outbound/public traffic,
executes as `user`, and deletes the VM before reporting success. Files persist
through the session mount; VM processes and memory are not reused. Cancellation
during creation waits for the create response so the resulting VM can still be
deleted. Failed deletion is recorded as `destroy_failed`, never a successful
cleanup. A finite server idle timeout uses `on_timeout=kill`, without auto-resume;
it supplements cleanup and is not claimed to be a hard lifetime guarantee while
there is continuing activity. A per-run lifecycle receipt is stored outside the
mounted workspace. API keys are not injected into the VM.

Official deployment references:

- https://docs.cubesandbox.com/zh/guide/quickstart
- https://docs.cubesandbox.com/zh/guide/pvm-deploy
- https://github.com/TencentCloud/CubeSandbox/blob/master/sdk/python/README.md

Without native KVM, the official x86_64 PVM route requires administrator access,
installing a host kernel, boot configuration changes and a reboot. It also needs
the documented XFS storage and system services. These are host operations, not
Python dependencies. They have not been performed on the shared hl_gpu server.
The running previews retain their previous backend until real service deployment
and the live test below succeed; this does not claim CubeSandbox is running.

With those settings and the service ready, run:

```
PYTHONPATH=src VENUS_CUBE_LIVE_TEST=1 python -m pytest -q tests/harness/test_session_cube_live.py
```

`test_session_cube.py` checks SDK/lifecycle contracts with controlled doubles.
The separate live test creates actual microVMs, verifies the session mount and
output persistence, and checks that the created IDs are absent after deletion.
It is skipped unless explicitly enabled and never falls back to Bubblewrap.

## Scientist fixes

Some compatible free models omit the `mcp__venusfactory__` prefix when returning
a function call. The evidence gateway now repairs an exact unambiguous alias
only when its canonical name was advertised in that request. Unknown functions
remain rejected. Raw provider evidence is retained; repairs are logged separately.
Fragmented streaming function names are assembled before validation.

Planning questions no longer automatically stop a study. The implementation
role can inspect resources and perform useful work first; an actual checkpoint
can still request indispensable user data. Critics must compare results with
the original objective, not approve a planner's substituted data.

Planning and review requests use JSON output. An invalid structured response
gets at most one formatting repair within the existing model-call budget,
instead of consuming a whole research iteration on Markdown/JSON mismatch.


## Compute allocation and sandbox memory

Local sandbox execution defaults to an 8 GiB address-space limit and a 4 GiB
host-memory reserve. The memory controller checks the smallest host/cgroup limit,
including parent cgroups. A PBS job or container limited to 8 GiB cannot satisfy
that configuration. It now fails immediately with the allocation and required
budget instead of waiting for a misleading "resources are busy" timeout.

Allocate sufficient memory for the sandbox, host reserve and controller process
(for example, 16 GiB for one default local sandbox), or set
`VENUS_SANDBOX_LOCAL_MEMORY_MB` for the actual workload. The platform does not
silently reduce sandbox limits or the host reserve. Temporary memory pressure and
other active reservations still wait in the admission queue. Online budgets,
concurrency limits and isolated execution remain enforced.
