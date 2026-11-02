import importlib
import subprocess
import sys
from pathlib import Path


def test_legacy_imports_share_canonical_identity():
    for old,new in [('model_gateway','engine.model_gateway'),('protein_references','engine.protein_references'),
                    ('case_plugins','campaigns.case_plugins'),('rfantibody_targets','cases.rfantibody_targets')]:
        assert importlib.import_module('agent.rsi.'+old) is importlib.import_module('agent.rsi.'+new)


def test_importing_rsi_does_not_eagerly_load_model_implementations():
    result=subprocess.run([sys.executable,'-c','import sys; import agent.rsi; assert "agent.rsi.engine.generated_model" not in sys.modules; assert "torch" not in sys.modules'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_source_snapshot_contains_subpackage_and_classified_scripts():
    from agent.rsi.engine.source_snapshot import research_implementation_paths
    root=Path(__file__).resolve().parents[2]
    names={p.relative_to(root).as_posix() for p in research_implementation_paths(root)}
    assert {'src/agent/rsi/_compat.py','src/agent/rsi/artifacts/package.py',
            'scripts/rsi/cases/run_clean_annotation.py','scripts/rsi/data_preparation/prepare_poisk.py'}<=names
