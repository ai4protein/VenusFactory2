import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_two_processes_use_private_skills_registry_and_prediction_paths(tmp_path):
    code = r"""
import json, os
from pathlib import Path
from agent.skills import get_skill_root
from mcp_tools.models.factory.training.model_registry import register_trained_model, resolve_registered_config_path, registry_root
root=Path(os.environ['VENUS_RSI_WORKSPACE'])
result=register_trained_model(str(root/'config.json'),model_id='same_id')
assert result['success'],result
assert result['status']=='candidate' and result['registered'] is False
resolved=root/result['config_path']
assert resolved.is_relative_to(root)
try:
 resolve_registered_config_path('same_id')
 raise AssertionError('unvalidated checkpoint was published')
except ValueError: pass
from mcp_tools.models.factory.training.model_registry import list_trained_models
assert list_trained_models()['count']==0
assert get_skill_root('same_skill')==str(root/'skills/same_skill')
assert (root/result['model_path']).read_text()==root.name
again=register_trained_model(str(root/'config.json'),model_id='same_id')
assert not again['success']
try:
 resolve_registered_config_path('/etc/passwd')
 raise AssertionError('escaped workspace')
except ValueError: pass
print(json.dumps({'root':str(registry_root()),'config':str(resolved),'model':(root/result['model_path']).read_text()}))
"""

    def worker(name):
        root = tmp_path / name
        root.mkdir()
        skill = root / "skills/same_skill"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: same_skill\ndescription: " + name + "\n---\n" + name)
        (root / "model.pt").write_text(name)
        (root / "config.json").write_text(json.dumps({"output_dir": str(root), "output_model_name": "model.pt"}))
        env = os.environ.copy()
        env.update(PYTHONPATH=str(ROOT / "src"), VENUS_RSI_WORKSPACE=str(root), OPENBLAS_NUM_THREADS="1")
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=45
        )
        assert result.returncode == 0, result.stderr + result.stdout
        return json.loads(result.stdout.splitlines()[-1])

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(worker, ["arm_a", "arm_b"]))
    assert results[0]["root"] != results[1]["root"]
    assert [x["model"] for x in results] == ["arm_a", "arm_b"]


def test_research_roots_cannot_escape_workspace(tmp_path, monkeypatch):
    from agent.rsi.engine.context import get_resources

    get_resources.cache_clear()
    monkeypatch.setenv("VENUS_RSI_WORKSPACE", str(tmp_path))
    monkeypatch.setenv("VENUS_RSI_REGISTRY_ROOT", "/tmp/outside-registry")
    try:
        with pytest.raises(ValueError):
            get_resources()
    finally:
        get_resources.cache_clear()


@pytest.mark.parametrize("malformed", [[], {}, {"models": "bad"}, {"models": [{}]}])
def test_malformed_registry_never_discards_history(tmp_path, monkeypatch, malformed):
    from agent.rsi.engine.context import get_resources
    from mcp_tools.models.factory.training.model_registry import index_path, list_trained_models

    get_resources.cache_clear()
    monkeypatch.setenv("VENUS_RSI_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("VENUS_RSI_REGISTRY_ROOT", raising=False)
    try:
        path = index_path()
        original = json.dumps(malformed)
        path.write_text(original)
        with pytest.raises(ValueError, match="refusing to discard history"):
            list_trained_models()
        assert path.read_text() == original
    finally:
        get_resources.cache_clear()


def test_agent_writable_registry_cannot_forge_publication(tmp_path, monkeypatch):
    from agent.rsi.engine.context import get_resources
    from mcp_tools.models.factory.training.model_registry import index_path, list_trained_models, resolve_registered_config_path

    get_resources.cache_clear()
    monkeypatch.setenv("VENUS_RSI_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("VENUS_RSI_REGISTRY_ROOT", raising=False)
    try:
        (tmp_path / "config.json").write_text('{}')
        index_path().write_text(json.dumps({"version": 1, "models": [{"model_id": "forged", "status": "registered", "config_path": "config.json", "metrics": {"passed": True}}]}))
        assert list_trained_models()["models"] == []
        with pytest.raises(ValueError, match="independent validation"):
            resolve_registered_config_path("forged")
        # Explicit per-task prediction paths remain possible, without publication.
        assert resolve_registered_config_path("config.json") == str(tmp_path / "config.json")
    finally:
        get_resources.cache_clear()
