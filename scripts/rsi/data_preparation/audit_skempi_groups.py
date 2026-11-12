"""Audit mapped SKEMPI2 homologous partner groups locally; no model calls."""
import argparse
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(_root), str(_root / "src")]
from agent.rsi.datasets.skempi_protocol import homology_groups
from agent.rsi.datasets.skempi_source import canonical, read_regular, require, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.manifest).absolute()
    raw = read_regular(path, 1024**2)
    require(sha256(raw) == args.sha256, "Mapped manifest changed")
    manifest = json.loads(raw)
    require(manifest["schema"] == "skempi_source.v1" and manifest.get("structures"), "Mapped SKEMPI source required")
    data = read_regular(path.parent / "observations.jsonl", 64*1024**2)
    require(sha256(data) == manifest["observations_sha256"], "Mapped observations changed")
    rows = [json.loads(line) for line in data.decode().splitlines()]
    report = homology_groups(rows, progress=lambda event: print(json.dumps(event), flush=True))
    output = Path(args.output).absolute()
    require(not output.exists() and not any(p.is_symlink() for p in output.parents), "Use a new non-symlink output directory")
    output.mkdir(parents=True)
    body = "".join(canonical(row) + "\n" for row in rows).encode()
    (output / "observations.jsonl").write_bytes(body)
    report.update(source_manifest_sha256=args.sha256, observations_sha256=sha256(body))
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"edges", "candidate_group_sizes"}}, indent=2))


if __name__ == "__main__":
    main()
