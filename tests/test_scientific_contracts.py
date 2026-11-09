"""Released feedback identities can be checked without scientific dependencies."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BINDINGS = json.loads((ROOT / "tests/rsi/fixtures/context_feedback_abi.json").read_text())


@pytest.mark.parametrize("version", sorted(BINDINGS["policies"]))
def test_released_feedback_source_contract_is_preserved(version):
    # Stored Scientist feedback includes these source digests. A pure refactor
    # can change them and make already published models fail validation.
    for name, expected in BINDINGS["policies"][version].items():
        source = ROOT / "src/agent/rsi/engine" / name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, (
            f"Published group_baselines.v{version} source changed: {name}. "
            "Keep this version available and use a new policy for changed implementations."
        )
