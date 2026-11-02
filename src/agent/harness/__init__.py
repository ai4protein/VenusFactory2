"""Replaceable research execution adapters, independent of scientific mode and LLM."""

from .contracts import Budget, ModelBinding, Profile, Task
from .registry import HarnessRegistry, builtin_registry

__all__ = ["Budget", "ModelBinding", "Profile", "Task", "HarnessRegistry", "builtin_registry"]
