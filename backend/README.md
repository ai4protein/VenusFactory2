# API and task control

`api_server.py` provides FastAPI routes and the built React client. `web_v2/`
owns authentication, sessions, online policy, model catalog, durable task queue,
workers, event replay and result downloads. Agent research logic stays in
`src/agent/`; scientific tool implementations stay in `mcp_tools/`.

From the repository root:

```bash
python scripts/serve.py --mode local --component api --port 7861 --api-workers 2
```

Use the same environment, runtime root and MCP port for every process in a
deployment. See [deployment guide](../docs/architecture/services.md).
