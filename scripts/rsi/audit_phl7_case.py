"""Independent file and numerical replay of the public PHL7 OPES comparison.

Does not import the production PHL7 adapter, invoke PLUMED or call an LLM.
Every regenerated grid is checked directly against its pinned archive member.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np

ARCHIVE = "3dd9cb3d0bfcfa53d02ae528bdd940d3fc815e5570ba071f3f439daaabae9f91"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(campaign, archive, output):
    campaign, archive, output = (Path(p).resolve() for p in (campaign, archive, output))
    if output.exists():
        raise FileExistsError(output)
    if sha(archive) != ARCHIVE:
        raise ValueError("public archive changed")
    report = json.loads((campaign / "report.json").read_text())
    protocol = json.loads((campaign / "protocol.json").read_text())
    if report.get("archive_sha256") != ARCHIVE or protocol.get("archive_sha256") != ARCHIVE:
        raise ValueError("archive binding changed")
    if report["status"] != "completed" or report["protocol_sha256"] != sha(campaign / "protocol.json"):
        raise ValueError("campaign not complete or binding changed")
    if sha(campaign / "phl7_case.py") != protocol["module_sha256"] or sha(campaign / "run_phl7_case.py") != protocol["entry_sha256"]:
        raise ValueError("implementation snapshot changed")
    if report["new_molecular_dynamics_performed"] is not False or report["full_paper_reproduced"] is not False:
        raise ValueError("unsupported scope claim")
    if protocol["plumed_version"] != "2.9.0" or protocol["comparison_tolerances"] != {"coordinates": 1e-6, "free_energy_kj_mol": 5e-5, "derivatives": 1e-3}:
        raise ValueError("protocol changed")
    ids = [f"{a}/{v}/repl_{i}" for a in ("93_open_closed", "Met_rotation") for v in ("WT", "R4M6", "R4M9", "R4M10") for i in (1, 2, 3)]
    records = {x["id"]: x for x in report["allocations"]}
    if sorted(records) != sorted(ids) or len(report["allocations"]) != 24:
        raise ValueError("allocation denominator changed")
    results = []
    with zipfile.ZipFile(archive) as z:
        for ident in ids:
            analysis, variant, replica = ident.split("/")
            parent = "MD_simulations_PHL7/OPES/" + analysis + "/" + variant + "/"
            result_name = "fes-fh.dat" if analysis == "93_open_closed" else "fes.dat"
            folder = campaign / "allocations" / ident
            bindings = {"CV_": parent + replica + "/CV_", "reweight.dat": parent + "reweight.dat",
                        "reference.dat": parent + replica + "/" + result_name}
            if analysis == "93_open_closed":
                bindings["DD_"] = parent + replica + "/DD_"
            record = records[ident]
            if record["status"] != "completed" or record["exit_code"] != 0:
                raise ValueError("incomplete PLUMED execution")
            if set(record["artifact_hashes"]) != {*bindings, result_name, "driver.log"}:
                raise ValueError("missing input/output artifacts")
            for filename, member in bindings.items():
                if (folder / filename).read_bytes() != z.read(member):
                    raise ValueError("analysis input differs from archive")
            for name, expected in record["artifact_hashes"].items():
                if sha(folder / name) != expected:
                    raise ValueError("artifact changed")
            ref_bytes = z.read(bindings["reference.dat"])
            actual_bytes = (folder / result_name).read_bytes()
            ref = np.loadtxt(io.BytesIO(ref_bytes))
            actual = np.loadtxt(io.BytesIO(actual_bytes))
            if ref.shape != actual.shape or ref.shape[1] != 5 or np.isnan(actual).any() or np.isneginf(actual).any():
                raise ValueError("invalid regenerated grid")
            if not np.isfinite(actual[:, :2]).all() or not np.array_equal(ref[:, :2], actual[:, :2]):
                raise ValueError("grid geometry changed")
            mask = np.isfinite(ref[:, 2:]) & np.isfinite(actual[:, 2:])
            errors = [float(np.max(np.abs(ref[mask[:, j], j + 2] - actual[mask[:, j], j + 2]))) for j in range(3)]
            support_equal = bool(np.array_equal(np.isfinite(ref[:, 2:]), np.isfinite(actual[:, 2:])))
            matches = support_equal and errors[0] <= 5e-5 and max(errors[1:]) <= 1e-3
            declared = record["comparison"]
            observed = [declared["free_energy_max_abs_error"], *declared["derivative_max_abs_errors"]]
            if any(not math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12) for a, b in zip(observed, errors)) or declared["matches_reference"] is not matches:
                raise ValueError("reported comparison inconsistent with direct archive replay")
            if declared["grid_rows"] != len(ref) or declared["finite_support_identical"] is not support_equal:
                raise ValueError("reported grid scope inconsistent")
            results.append({"id": ident, "grid_rows": len(ref), "bitwise_equal": actual_bytes == ref_bytes,
                            "finite_support_equal": support_equal, "max_abs_errors": errors, "numeric_match": matches})
    result = {"status": "passed", "scope": "Independent comparison of every regenerated grid to the original pinned archive; does not regenerate MD.",
              "archive_sha256": sha(archive), "report_sha256": sha(campaign / "report.json"),
              "protocol_sha256": sha(campaign / "protocol.json"), "auditor_sha256": sha(Path(__file__)),
              "allocations": results, "bitwise_matches": sum(x["bitwise_equal"] for x in results),
              "numeric_matches": sum(x["numeric_match"] for x in results), "full_paper_reproduced": False}
    output.mkdir(parents=True)
    shutil.copy2(Path(__file__), output / "auditor.py")
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.campaign, args.archive, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "allocations"}, indent=2))
