"""Recompute terminal REAP split and metric evidence without refitting models."""
from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run, admission, output):
    report = json.loads((run / "report.json").read_text())
    protocol = json.loads((run / "protocol.json").read_text())
    rows = json.loads((admission / "campaign.json").read_text())
    if report["status"] != "completed" or len(report["fits"]) != 100 or len(protocol["allocations"]) != 100:
        raise ValueError("All 100 terminal allocations required")
    if report["protocol_sha256"] != sha(run / "protocol.json") or protocol["input_sha256"] != sha(admission / "campaign.json"):
        raise ValueError("Protocol or inputs changed")
    plans = protocol["allocations"]
    if len({p["id"] for p in plans}) != 100:
        raise ValueError("Duplicate allocation")
    y = np.asarray([x["target"] for x in rows], dtype=np.float64)
    checked, groups, partitions = [], {}, {}
    for planned, recorded in zip(plans, report["fits"], strict=True):
        if any(planned[k] != recorded[k] for k in ["id", "family", "seed", "loss"]):
            raise ValueError("Allocation identity changed")
        tr, va, te = [planned[k] for k in ["train", "validation", "future"]]
        expected_single = {i for i,x in enumerate(rows) if x["family"] == planned["family"] and x["mutation_order"] == 1}
        expected_future = {i for i,x in enumerate(rows) if x["family"] == planned["family"] and x["mutation_order"] > 1}
        if set(tr) | set(va) != expected_single or set(te) != expected_future or set(tr) & set(va) or any(len(v) != len(set(v)) for v in [tr,va,te]):
            raise ValueError("Declared split coverage differs")
        sequences = [{rows[i]["sequence"] for i in v} for v in [tr,va,te]]
        if any(sequences[i] & sequences[j] for i in range(3) for j in range(i)):
            raise ValueError("Sequence leakage")
        split_key = (planned["family"], planned["seed"])
        split = (tr,va,te)
        if split_key in partitions and partitions[split_key] != split:
            raise ValueError("Unequal method splits")
        partitions[split_key] = split
        mu, sd = float(y[tr].mean()), float(y[tr].std())
        if not np.allclose([mu,sd], [recorded["target_mean"],recorded["target_std"]], rtol=1e-12,atol=1e-12):
            raise ValueError("Training-only standardization changed")
        predictions = run / planned["id"] / "predictions.npz"
        if sha(predictions) != recorded["prediction_sha256"]:
            raise ValueError("Saved predictions changed")
        with np.load(predictions, allow_pickle=False) as saved:
            if saved["future_indices"].tolist() != te or not np.array_equal(saved["observed"],y[te]):
                raise ValueError("Future identity or truth changed")
            values = saved["predicted_standardized"].astype(np.float64)
            truth = ((y[te]-mu)/sd).astype(np.float32).astype(np.float64)
            mse = float(np.square(values-truth).mean())
            rx,ry = rankdata(values),rankdata(truth)
            rho = 0. if min(rx.std(),ry.std()) == 0 else float(np.corrcoef(rx,ry)[0,1])
            if not np.allclose([mse,rho],[recorded["future_standardized_mse"],recorded["future_spearman"]],rtol=1e-6,atol=1e-7):
                raise ValueError("Independently calculated future metric differs")
            if not np.allclose(saved["predicted"],values*sd+mu,rtol=1e-6,atol=1e-6):
                raise ValueError("Prediction units differ")
        checked.append({"id":planned["id"],"mse":mse,"spearman":rho})
        groups.setdefault((planned["family"],planned["loss"]),[]).append(rho)
    summary=[{"family":f,"method":m,"n_seeds":len(v),"spearman_mean":float(np.mean(v)),"spearman_sd":float(np.std(v,ddof=1))} for (f,m),v in groups.items()]
    result={"status":"passed","report_sha256":sha(run/'report.json'),"protocol_sha256":sha(run/'protocol.json'),"auditor_sha256":sha(Path(__file__)),"allocations_checked":len(checked),"metrics":checked,"summary":summary,"scope":"Independent saved-prediction metric and split audit; no independent feature pass or model refit. Seeds are algorithmic repetitions, not biological replicates. No claim of exact author table agreement, autonomous orchestration or RSI gain."}
    output.mkdir(parents=True,exist_ok=False)
    (output/'audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (output/'auditor.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','admission','output']:
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    audit(args.run,args.admission,args.output)
