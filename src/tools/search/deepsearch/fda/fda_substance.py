#!/usr/bin/env python3
"""
FDA substance/other: UNII lookup, substance by name.
Uses FDAQuery from fda_query. Skill: src/agent/skills/fda/

Run: python -m src.tools.search.deepsearch.fda.fda_substance
"""

import os

try:
    from .fda_query import FDAQuery
except ImportError:
    from fda_query import FDAQuery


def main():
    """Run substance-related openFDA examples."""
    api_key = os.environ.get("FDA_API_KEY")
    fda = FDAQuery(api_key=api_key)

    print("openFDA substance endpoints")
    print("=" * 50)

    print("\n1. query_substance_by_name('ibuprofen')...")
    out = fda.query_substance_by_name("ibuprofen")
    if "results" in out:
        print(f"   Found {len(out['results'])} substance(s)")
        if out["results"]:
            r = out["results"][0]
            print("   First approvalID:", r.get("approvalID"))
    else:
        print("   ", out.get("error", "No results"))

    print("\n2. query_substance_by_unii (example UNII)...")
    out = fda.query_substance_by_unii("R16CO5Y76E")
    if "results" in out and out["results"]:
        print("   Substance by UNII retrieved")
    else:
        print("   No result for UNII")
    print("\nDone.")


if __name__ == "__main__":
    main()
