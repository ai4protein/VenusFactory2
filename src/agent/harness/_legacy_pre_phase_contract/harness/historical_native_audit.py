"""Dispatch only explicitly pinned historical native-Scientist audit code.

The archived verifier is immutable and checks the original evidence itself.
Unknown revisions fail closed; this is not a loader for code from run folders.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from .contracts import file_hash

PINNED_SOURCES = {
    "__init__.py": "6672043a6f8d56a39853beaadb260264e368a43a706122b72c55f9f80a01cd33",
    "harness/__init__.py": "6672043a6f8d56a39853beaadb260264e368a43a706122b72c55f9f80a01cd33",
    "harness/contracts.py": "7bab4b5b31f80479e7d337bfad81017c9e6ad6193bb1a895ce4d3b18e8e2158d",
    "harness/empty_response_recovery.py": "a84a5c06220ed8c771e8e15aa71650a5e2ce134ff824440273bdfeab32311f58",
    "harness/events.py": "919845fc18aab0265ea9d5c30050fc7ab40afa1788f37a55edff1aba073c3773",
    "harness/host.py": "61eda657cf06d73a3cf8368bfe6b7832cc613f2184620db002e9ed0f410cfdb3",
    "harness/native_expert_audit.py": "92619ce644830728557cd48ca08930bdb0869e333f65217f5ac475515c599f5e",
    "harness/native_expert_team.py": "c263bd91f399dff1bf3225a54d39eaa97bebcd343663b87aaf5fc6b8328af11b",
    "harness/registry.py": "d04d03adde2ad178debad6e67a4e51bd60b3b6d6a4c47adf58c63096c81b192e",
    "harness/request_recovery.py": "4655839ae9fd490444833c79a852d738c461b76e1f78f6f3eb4e0792ee6a6997",
    "rsi/__init__.py": "6672043a6f8d56a39853beaadb260264e368a43a706122b72c55f9f80a01cd33",
    "rsi/model_gateway.py": "6205d4331aa068eee82ca06248a34c72cef671c6f795a268e154a9b97961cf02",
    "rsi/scientific_episode.py": "65467a5a8c7397d6cfd625f2cdc26af49dc7dd070eddcdd90bc91099ed57a4f2",
}


def historical_auditor(lock):
    if lock.get("adapter_source", {}).get("native_expert_team.py") != PINNED_SOURCES["harness/native_expert_team.py"]:
        from .historical_tool_audit import historical_tool_auditor

        return historical_tool_auditor(lock)
    profile = lock["profile"]
    if (
        profile.get("mode") != "science_expert"
        or profile.get("harness") not in {"expert-kimi-code", "expert-deepseek-harness"}
        or profile.get("model", {}).get("protocol", "chat_completions") != "chat_completions"
    ):
        raise ValueError("No pinned historical auditor for this profile")
    for name in ("native_expert_team.py", "contracts.py", "events.py", "native_expert_audit.py"):
        if lock.get("adapter_source", {}).get(name) != PINNED_SOURCES["harness/" + name]:
            raise ValueError("Unrecognized historical expert source")
    for name in ("rsi/model_gateway.py", "rsi/scientific_episode.py"):
        expected = PINNED_SOURCES[name]
        if lock.get("supporting_source", {}).get(name) != expected:
            raise ValueError("Unrecognized historical supporting source")
        if file_hash(Path(__file__).parent.parent / name) != expected:
            raise ValueError("Historical import support changed")
    root = Path(__file__).parent / "_legacy_native_v2"
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Missing historical verifier capsule")
    actual = {str(path.relative_to(root)) for path in root.rglob("*.py")}
    if actual != set(PINNED_SOURCES):
        raise ValueError("Historical verifier capsule layout changed")
    for name, expected in PINNED_SOURCES.items():
        path = root / name
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root.parent):
            raise ValueError("Symlinked historical verifier source")
        if file_hash(path) != expected:
            raise ValueError("Historical verifier source changed")
    module = importlib.import_module("agent.harness._legacy_native_v2.harness.native_expert_audit")
    return module.audit_native_expert
