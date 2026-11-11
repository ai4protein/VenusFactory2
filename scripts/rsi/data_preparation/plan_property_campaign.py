"""Prepare a source-bound, property-balanced allocation without model calls."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from agent.rsi.campaigns.property_coverage import validate_policy
from agent.rsi.campaigns.shared_acquisition_plan import plan_acquisition
from agent.rsi.datasets.research_project_catalog import build_catalog
from agent.rsi.engine.storage import sha256_file


def reference(path):
    path = path.absolute()
    return {"path": str(path), "sha256": sha256_file(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-policy", type=Path, default=ROOT / "configs/rsi/property_coverage_v1.json")
    parser.add_argument("--projects", nargs="+")
    parser.add_argument("--exclude-panel", action="append", type=Path, default=[])
    parser.add_argument("--registered-catalog", action="append", type=Path, default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    manifest_ref, policy_ref = reference(args.manifest), reference(args.coverage_policy)
    manifest = json.loads(args.manifest.read_text())
    policy = json.loads(args.coverage_policy.read_text())
    target_count = validate_policy(policy)
    catalog = build_catalog(manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    catalog_path = args.output / "catalog.json"
    catalog_path.write_text(json.dumps(catalog, indent=2) + "\n")
    allocation = plan_acquisition(
        manifest_ref,
        reference(catalog_path),
        projects=args.projects or [p["id"] for p in manifest["projects"]],
        target_count=target_count,
        batch_size=args.batch_size,
        coverage_policy=policy_ref,
        previous_panels=[reference(p) for p in args.exclude_panel],
        registered_catalogs=[reference(p) for p in args.registered_catalog],
    )
    for name, value in [("allocation.json", allocation), ("coverage.json", allocation["coverage"])]:
        (args.output / name).write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": allocation["status"],
                "requested": target_count,
                "selected": allocation["selected_count"],
                "families": allocation["coverage"]["families"],
                "external_model_calls": 0,
                "new_autonomous_models": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
