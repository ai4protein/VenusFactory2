# Run Scientist on a local cluster

A local installation can submit a complete Scientist goal to an operator-configured
PBS or Slurm cluster. The scheduler allocates resources; the existing Scientist
controller preserves the goal, source snapshot, budget, execution identity and
results. Online deployments cannot load cluster configurations or submit, query,
release or cancel cluster work through these interfaces.

Use the CLI on a cluster login node with the same Unix account and shared
filesystem as its compute nodes. The checkout, goal directory, Python environment
and installed runtimes must be accessible at the same paths on both. This is not
an arbitrary SSH-command interface. The Web Scientist launcher continues to use
its existing execution path; it does not automatically submit topics to a cluster.

## Choose an execution profile

| Scheduler | Current execution support | Verification |
| --- | --- | --- |
| PBS | One-node, one-GPU allocation; CPU or verified GPU fits within it | Exercised on the development PBS cluster |
| Slurm | One-node CPU allocation with enforced CPU and memory limits | Command, lifecycle and isolation fixtures; validate on your Slurm site before use |

Slurm GPU execution is not enabled. A scheduler allocation or a CUDA environment
variable alone does not establish device isolation. CPU Slurm jobs cannot invoke
GPU model jobs or request CUDA fitting through MCP.

Copy [the cluster example](../configs/cluster-goals.example.json) to a private
configuration such as `configs/cluster-goals.local.json`. For Slurm, set:

- `scheduler.backend` to `slurm`, `scheduler.cluster` to the actual cluster name,
  and `scheduler.bin_directory` to its installed command directory.
- `profiles.cpu.queue` to the partition. Choose 1–8 CPUs, zero GPUs, 16–64 GiB
  memory and a whole-minute wall time up to one day.
- `python` to the installed Python environment and `runtime_root` to the project
  installation holding its optional tools and harnesses.
- `repository` to a clean, committed checkout, `control_root` to private local
  state, and `goal_roots` to the directories this installation may execute.

Relative paths resolve from the configuration file's directory. Executable paths
inside `environment` must be absolute. Use a self-contained environment for
bubblewrap, including its Python standard library and training dependencies.
Keep credentials in the protected `env_file` or credential file; do not put key
values in scheduler settings. Only selected admitted credentials reach trusted
processes, not model-authored code.

For PBS, the existing [PBS configuration](../configs/pbs-goals.example.json) and
`submit-pbs`, `status-pbs`, `cancel-pbs` commands remain compatible. The new schema
also accepts `scheduler.backend: "pbs"`; its `cluster` is the administrator's
stable label for that configured local endpoint. PBS profiles require one GPU.

## Submit and inspect a goal

Create a goal using the [Scientist CLI](scientist-model-development.md#submit-a-complete-scientist-goal-to-pbs),
or use an existing prepared goal. Then, from the repository root:

```bash
export WEBUI_V2_MODE=local
export VENUS_SCIENTIST_ONLINE=0
python scripts/rsi/run_autonomous_research.py submit-cluster \
  --output runtime/local/scientist-goals/my-topic \
  --cluster-config configs/cluster-goals.local.json --resource-profile cpu
python scripts/rsi/run_autonomous_research.py status-cluster \
  --output runtime/local/scientist-goals/my-topic \
  --cluster-config configs/cluster-goals.local.json
python scripts/rsi/run_autonomous_research.py cancel-cluster \
  --output runtime/local/scientist-goals/my-topic \
  --cluster-config configs/cluster-goals.local.json
```

Submission first creates a held job. The controller verifies its owner, unique
attempt identifier and resource request before release. Status reconciles the
same saved job; if a command times out or accounting is temporarily unavailable,
its state remains unknown rather than triggering another submission. No local
process is kept waiting for each queued allocation.

A new attempt requires `--new-attempt`; add `--resume-goal` if the saved goal needs
to resume. These flags do not reset its research budget. Active cluster ownership
prevents a second direct `run` or `resume`. Cancellation verifies the original
job identity; it does not cancel jobs merely because they share a name.

A running worker validates its scheduler identity and enforced allocation before
loading credentials or starting research. CPU Slurm requires verifiable cgroup
CPU and memory limits in a verified `job_<id>` cgroup layout. SLUID-only layouts
need additional site integration; unsupported or incomplete evidence stops execution. It
does not fall back to unrestricted execution on the login host. Slurm installations
must support `sbatch --export=NIL`, non-requeued jobs, `scontrol` and allocation
accounting through `sacct`.

The worker starts its own bounded MCP queue and cleans up its services and child
jobs on exit. Research completion is determined by SC and Review, not the
scheduler's exit code. Failed attempts and partial results remain available.

## Code boundaries

- `backend/compute/` defines local admission and bounded scheduler adapters.
- The Scientist controller owns durable attempts, source snapshots and goal state.
- The MCP queue checks the admitted worker and resource scope before tool execution.

The adapters do not receive prompts or arbitrary shell scripts. They launch a
fixed worker with host-selected paths. Existing task records remain readable;
there is no second research queue or copied task state machine.
