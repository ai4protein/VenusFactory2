"""Verifier for the exact pre-tool-policy native Scientist source revision."""

import importlib
from pathlib import Path

from .contracts import file_hash

PINNED = {
    "__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "harness/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "harness/adapters.py": "e0d5c3eb50c00c45e5cfc15ceecc2ee16964071e284c43c4c1752402111d544b",
    "harness/catalog.py": "8929d6b8d57951eca0897ec564e7c76528d0391b3dc183ba3020d36ebbd28ca5",
    "harness/codex_adapter.py": "5b62f5262c7652d5553cb6aafe3b4e582f8a933fe0a89ed88d0ff457332f4cd9",
    "harness/contracts.py": "c740a7dcd40220ace1e44d9b95c3dda191d11247f1f1714cc0226dfe14caecb7",
    "harness/dsh_worker.py": "87c7c7af3b7a97a3e7c8393cbb5c50da31c1b20202b9cfdf5305ef56d6d05bea",
    "harness/empty_response_recovery.py": "a84a5c06220ed8c771e8e15aa71650a5e2ce134ff824440273bdfeab32311f58",
    "harness/events.py": "919845fc18aab0265ea9d5c30050fc7ab40afa1788f37a55edff1aba073c3773",
    "harness/gateway.py": "736ef8bd415640a2caae811f76bff92466e6e58f70998330059739c6d7fa823a",
    "harness/historical_native_audit.py": "dbfe5ed3b60fa1bfaaec7f93598887bc0c28b0c277f0fbab283c8beca80bfa67",
    "harness/host.py": "76651522d67555f3212ac3c07c165acd07b74b211ad7a55003777b2dffa7cff2",
    "harness/native_expert_audit.py": "9ba41737719348a98653f8de7e55f48e70cacffff702063f5bc25ae5f36f428b",
    "harness/native_expert_team.py": "0d2f1eaf96b0df0a1f3d1c89f38e5509e44f1d7aa2da1553cb80455882582a2a",
    "harness/registry.py": "45a19094bbeb9f1324d906f5f328e85c0703a2fce5f23cf5082ba40fbe54efb4",
    "harness/request_recovery.py": "4655839ae9fd490444833c79a852d738c461b76e1f78f6f3eb4e0792ee6a6997",
    "harness/responses_gateway.py": "417f1e0eca36edca1fdef0d2c9209931ae1441f4454ae11c901e314a8fd63c25",
    "harness/web_langgraph_worker.py": "9defd2eecb45b75b435eb7db1a3e8afdc686ec846972e4a17a009f72c22b0522",
    "rsi/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "rsi/model_gateway.py": "6205d4331aa068eee82ca06248a34c72cef671c6f795a268e154a9b97961cf02",
    "rsi/scientific_episode.py": "65467a5a8c7397d6cfd625f2cdc26af49dc7dd070eddcdd90bc91099ed57a4f2",
}


def historical_tool_auditor(lock):
    for name in (
        "native_expert_team.py",
        "native_expert_audit.py",
        "contracts.py",
        "events.py",
        "responses_gateway.py",
        "codex_adapter.py",
    ):
        if lock.get("adapter_source", {}).get(name) != PINNED["harness/" + name]:
            raise ValueError("Unrecognized pre-tool-policy source")
    for name in ("rsi/model_gateway.py", "rsi/scientific_episode.py"):
        if (
            lock.get("supporting_source", {}).get(name) != PINNED[name]
            or file_hash(Path(__file__).parent.parent / name) != PINNED[name]
        ):
            raise ValueError("Historical tool-policy support changed")
    root = Path(__file__).parent / "_legacy_pre_tool_policy"
    if (
        root.is_symlink()
        or not root.is_dir()
        or {p.relative_to(root).as_posix() for p in root.rglob("*.py")} != set(PINNED)
    ):
        raise ValueError("Historical tool-policy capsule changed")
    for name, sha in PINNED.items():
        path = root / name
        if any(p.is_symlink() for p in (path, *path.parents)) or file_hash(path) != sha:
            raise ValueError("Historical tool-policy capsule changed")
    return importlib.import_module(
        "agent.harness._legacy_pre_tool_policy.harness.native_expert_audit"
    ).audit_native_expert
