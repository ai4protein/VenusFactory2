# Reference skills

`scientific-agent-skills/` is an intact Git submodule from
https://github.com/K-Dense-AI/scientific-agent-skills.git.

Initialize with `git submodule update --init --recursive`. The loader works with
native skills when the submodule is absent. Preserve the upstream LICENSE, commit
and package structure. Do not copy individual upstream skills into the native root.
`read_skill` loads reference content; upstream scripts are not automatically run.

Validate discovery from the repository root: `python scripts/validate_skills.py`.
