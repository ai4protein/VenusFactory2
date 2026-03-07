#!/usr/bin/env python3
"""
FDA device endpoints: adverse events, 510k, classification, enforcement.
Uses FDAQuery from fda_query. Skill: src/agent/skills/fda/

Run: python -m src.tools.search.deepsearch.fda.fda_device
"""

import os

try:
    from .fda_query import FDAQuery
except ImportError:
    from fda_query import FDAQuery


def main():
    """Run device-related openFDA examples."""
    api_key = os.environ.get("FDA_API_KEY")
    fda = FDAQuery(api_key=api_key)

    print("openFDA device endpoints")
    print("=" * 50)

    print("\n1. query_device_events('pacemaker', limit=5)...")
    out = fda.query_device_events("pacemaker", limit=5)
    if "results" in out:
        print(f"   Found {len(out['results'])} events")
    else:
        print("   ", out.get("error", "No results"))

    print("\n2. query_device_510k(applicant='Medtronic')...")
    out = fda.query_device_510k(applicant="Medtronic")
    if "results" in out:
        print(f"   510(k) clearances: {len(out['results'])}")
    else:
        print("   No results")

    print("\n3. query_device_classification('DQY')...")
    out = fda.query_device_classification("DQY")
    if "results" in out and out["results"]:
        print("   Classification retrieved")
    else:
        print("   No classification")
    print("\nDone.")


if __name__ == "__main__":
    main()
