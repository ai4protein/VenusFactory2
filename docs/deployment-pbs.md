# Run online and local services with PBS

Build the frontend before submitting a job, using the supported Node 24 environment:

```bash
cd frontend
npm ci
npm test
npm run build
cd ..
```

Use a tested Python environment for the service. Keep optional model runtimes and
weights under `runtime/`; do not install or download them during every PBS launch.
For bubblewrap execution, use a self-contained Conda prefix (including its Python
standard library), rather than a virtualenv whose interpreter points outside the
mounted prefix. The PBS template contains no host paths or credentials:

```bash
qsub -q YOUR_QUEUE -v VENUS_SERVICE_PYTHON="$CONDA_PREFIX/bin/python" scripts/examples/factory_web.pbs
```

The job supervises both modes and exits if a managed service fails. There is no
interactive shell or fixed sleep after the web process exits. On termination it
stops its own API processes, MCP services and recorded descendants. PBS resource
limits cover the complete allocation. Both modes share the allocated hardware;
use separate jobs/allocations when they need independent GPU capacity.

| Mode | API | MCP | State and logs |
| --- | --- | --- | --- |
| Online | `127.0.0.1:7863` | `127.0.0.1:8085` | `runtime/online/` |
| Local | `127.0.0.1:7862` | `127.0.0.1:8084` | `runtime/local/` |

The launcher explicitly separates session databases, queues, quotas, usage,
uploads, reports, models and Scientist goals. Online restrictions remain enforced
by the backend. Local mode stays on loopback: access it through SSH, not a public
reverse proxy. Forward the online API through your existing HTTPS proxy.

```bash
python scripts/deploy_services.py --mode both
python scripts/deploy_services.py --mode online --online-port 7863
python scripts/deploy_services.py --mode local --local-port 7862
```

For an existing FRP installation, pass `--frp-executable /path/to/frpc` and
`--frp-config /private/path/to/frpc.ini`. Only the online port belongs in the public
web route. Keep the configuration outside Git and readable only by the service
owner. FRP starts after API and MCP readiness checks succeed.

Kimi configuration and daemon state live in `runtime/{online,local}/control/kimi`.
Existing provider settings are copied once with private permissions; global
conversation history and MCP declarations are not copied. The local Kimi daemon
uses port 58628 (override with `--local-kimi-port`) and cannot reuse the other
installation's daemon.

Read `runtime/services/deployment.json` for the PBS job, node, ports and readiness.
API output goes to `runtime/{online,local}/logs/web.log`; the tool service keeps its
log under the mode's queue directory. Do not start another bundle while the
deployment lock is held. Stop the old allocation before replacing it:

```bash
qstat -f JOB_ID
qdel JOB_ID
qsub -q YOUR_QUEUE -v VENUS_SERVICE_PYTHON="$CONDA_PREFIX/bin/python" scripts/examples/factory_web.pbs
```

Updates should pass source/secret checks, dependency auditing, API security tests
and the frontend build before replacing the running job. Back up private state
and record the deployed commit. Generated logs, environment files, databases and
dependency environments remain ignored by Git.
