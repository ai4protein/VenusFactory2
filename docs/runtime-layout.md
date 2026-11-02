# Local and online runtime directories

The application uses one source checkout and two independent data directories:

```
runtime/
  local/
    state/       # sessions, access policy, analytics and signing key
    files/       # uploads, conversation workspaces and generated files
    queue/       # durable jobs, events, workers and MCP service control
    scientist/   # goals, plans and checkpoints
    reports/     # Researcher jobs and reports
    agent/       # Agent execution evidence
    models/      # custom datasets, models, predictions and workflows
  online/        # same layout; separate mutable state
```

Pretrained weights, public datasets and source code remain shared. Online
ownership checks, quotas, upload and compute policies still apply; separating
folders does not replace authorization.

Launch from the repository root:

```
python backend/webui_v2.py --port 7862 --runtime-root ./runtime
python backend/webui_v2.py --online --port 7863 --runtime-root ./runtime
```

`VENUS_RUNTIME_ROOT` sets the common base. The application always appends `local`
or `online`. Environment variables for individual services override their default
paths, preserving existing installations. Do not point those overrides at the
same mutable directory when running both modes. The default MCP ports are 8080
and 8081; assign different `MCP_HTTP_PORT` values when other services use them.

Existing databases must be copied with SQLite's backup API while no new work is
being submitted. Preserve signing keys with the session database. Copy historical
files and records without deleting the original directories or rewriting scientific
evidence. Historical artifact paths may retain their original locations for
compatibility; new work uses the selected mode's paths. Do not copy live worker
control files or replay historical queued commands into a second service.

The current queue is a single-node SQLite implementation with process leases and
filesystem locks. It is not a PostgreSQL or distributed-worker implementation.
