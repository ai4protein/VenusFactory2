#!/usr/bin/env python3
"""
FDA drug endpoints: adverse events, labeling, NDC, enforcement (recalls), shortages.
Uses FDAQuery from fda_query. Skill: src/agent/skills/fda/

Run: python -m src.tools.search.deepsearch.fda.fda_drug
"""

import os

try:
    from .fda_query import FDAQuery
except ImportError:
    from fda_query import FDAQuery


def main():
    """Run drug-related openFDA examples."""
    api_key = os.environ.get("FDA_API_KEY")
    fda = FDAQuery(api_key=api_key)

    print("openFDA drug endpoints")
    print("=" * 50)

    print("\n1. query_drug_events('metformin', limit=5)...")
    out = fda.query_drug_events("metformin", limit=5)
    if "results" in out:
        print(f"   Found {len(out['results'])} events")
    else:
        print("   ", out.get("error", "No results"))

    print("\n2. query_drug_label('Keytruda', brand=True)...")
    out = fda.query_drug_label("Keytruda", brand=True)
    if "results" in out and out["results"]:
        print("   Label retrieved")
    else:
        print("   No label")

    print("\n3. query_drug_recalls(drug_name='metformin')...")
    out = fda.query_drug_recalls(drug_name="metformin")
    if "results" in out:
        print(f"   Recalls: {len(out['results'])}")
    else:
        print("   No recalls")
    print("\nDone.")


if __name__ == "__main__":
    main()
