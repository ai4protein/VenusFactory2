"""Load the complete application in a fresh process, with no model or network calls."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("mode", ["online", "local"])
def test_complete_api_loads_and_serves_workspace_contracts(tmp_path, mode):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, WEBUI_V2_MODE=mode, PYTHON_DOTENV_DISABLED="1",
               VENUS_RUNTIME_ROOT=str(tmp_path / "runtime"),
               WEBUI_V2_FRONTEND_DIST=str(tmp_path / "frontend"),
               VENUS_MODEL_CONFIG=str(root / "model-config.yaml.example"),
               VENUS_MODEL_RUNTIMES_CONFIG=str(tmp_path / "absent-runtimes.json"),
               KIMI_CODE_HOME=str(tmp_path / "kimi"), VENUS_WORKER_AUTOSTART="0",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    # Overrides inherited from an operator's shell must not reach real stores.
    from backend.web_v2.runtime_paths import AREAS
    for name in AREAS:
        env.pop(name, None)
    code = r"""
import json, socket, sys
from fastapi.testclient import TestClient

def no_network(*args, **kwargs):
    raise AssertionError('Application configuration must not contact external services')
socket.socket.connect = no_network
from backend.api_server import app
from backend.web_v2.quick_tools_api import MutationRunBody
from pydantic import ValidationError
# Do not enter lifespan: MCP, harness daemons and inference stay stopped.
client = TestClient(app)
paths = ['/health', '/api/chat/agent-execution-options',
         '/api/chat/science-goals/catalog?view=composer', '/api/report/options',
         '/api/model-platform/catalog?limit=1', '/api/model-platform/workflow-components?limit=1']
for path in paths:
    response = client.get(path)
    assert response.status_code == 200, (path, response.status_code, response.text[:1000])
    assert isinstance(response.json(), dict)
detail = client.get('/api/model-platform/workflow-components/tool:query_pubmed')
assert detail.status_code == 200 and 'query' in detail.json()['contract']['properties']
assert not {'backend', 'api_key'} & MutationRunBody.model_json_schema()['properties'].keys()
for field in ('backend', 'api_key'):
    try:
        MutationRunBody.model_validate({'sequence': 'ACDE', field: 'invalid-host-option'})
    except ValidationError:
        pass
    else:
        raise AssertionError('Obsolete mutation options must be rejected')
assert MutationRunBody(sequence='ACDE', model_name='ProSST-2048').model_name == 'ProSST-2048'
assert not {'torch', 'transformers'} & set(sys.modules), 'API catalogs must not load inference runtimes'
print(json.dumps({'checked': paths}))
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=root, env=env,
                            capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
