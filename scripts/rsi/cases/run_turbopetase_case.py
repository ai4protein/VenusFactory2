"""Reproduce author TurboPETase inference and separately audit residue indexing.

Run with research_runs/rsi/envs/turbopetase/bin/python. Original author model
pickles are loaded only after checking the pinned asset manifest hashes.
"""

from __future__ import annotations

# Direct-script entry point; implementation imports remain package-qualified.
import sys as _entry_sys
from pathlib import Path as _EntryPath
_entry_root = _EntryPath(__file__).resolve().parents[3]
_entry_sys.path[:0] = [str(_entry_root), str(_entry_root / 'src'), str(_entry_root / 'scripts/rsi')]


import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
COMMIT = "b83bbfb3298176465d43e0803f8ba810b183c0bd"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def author_sequence(notebook):
    cells = json.loads(notebook.read_text())["cells"]
    source = "\n".join("".join(c.get("source", [])) for c in cells if c["cell_type"] == "code")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "raw_sequence" for t in node.targets
        ):
            sequence = ast.literal_eval(node.value)
            if not isinstance(sequence, str) or not set(sequence) <= set("ACDEFGHIKLMNPQRSTVWY"):
                raise ValueError("invalid author sequence")
            return sequence, source
    raise ValueError("author sequence absent")


def score(model, sequence, vocab, device, residue_offset):
    tokens = [2] + [vocab[aa] for aa in sequence] + [3]
    scores = []
    for i, aa in enumerate(sequence):
        inp = torch.tensor([tokens], device=device)
        pos = i + residue_offset
        inp[0, pos] = 4
        with torch.inference_mode():
            logits = model(inp)[0, pos].cpu().numpy()
        scores.append((logits - logits[vocab[aa]])[5:])
    matrix = np.asarray(scores)
    if matrix.shape != (len(sequence), 20) or not np.isfinite(matrix).all():
        raise ValueError("invalid mutation score matrix")
    return matrix


def main(args):
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.source
    required = [
        source / name for name in ["mlm_data_analysis.ipynb", "vocab.txt", "exp_data.txt", "mlm.py", "README.md"]
    ]
    weights = [source / "models" / (name + "_19.pt") for name in args.models]
    # The transfer manifest may use a list of records or a path-keyed mapping.
    manifest = json.loads((source / "asset_manifest.json").read_text())
    manifest_text = json.dumps(manifest)
    for path in required + weights:
        if sha(path) not in manifest_text:
            raise ValueError("asset does not match pinned download manifest: " + path.name)
    (args.output / "implementation.py").write_bytes(Path(__file__).read_bytes())
    sequence, notebook_source = author_sequence(source / "mlm_data_analysis.ipynb")
    vocab = {word: i for i, word in enumerate((source / "vocab.txt").read_text().splitlines())}
    amino_acids = list(vocab)[5:]
    if len(amino_acids) != 20 or set(amino_acids) != set(sequence):
        raise ValueError("vocabulary mismatch")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    report = {
        "status": "running",
        "repository": "https://github.com/Wublab/code-for-TurboPETase",
        "commit": COMMIT,
        "assets": {str(p.relative_to(source)): sha(p) for p in required + weights},
        "environment": {
            p: importlib.metadata.version(p) for p in ["torch", "numpy", "x-transformers", "entmax", "einops"]
        },
        "device": str(device),
        "sequence_length": len(sequence),
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "author_numbering_offset": 36,
        "position_selection": "author mean over 20 amino-acid logit differences; reported positions < 280",
        "comparisons": [],
        "agent_orchestrated": False,
        "claim_scope": "author released checkpoint inference; separate indexing sensitivity analysis; not retraining, wet-lab or full-paper replication",
    }
    json_save(args.output / "report.json", report)
    for weight in weights:
        model = torch.load(weight, map_location=device, weights_only=False).eval()
        for offset, label in [(0, "author_notebook"), (1, "residue_index_corrected")]:
            matrix = score(model, sequence, vocab, device, offset)
            prefix = weight.stem + "_" + label
            np.save(args.output / (prefix + ".npy"), matrix, allow_pickle=False)
            with (args.output / (prefix + ".csv")).open("w") as f:
                writer = csv.writer(f)
                writer.writerow(["position", "wild_type", "mutant", "score"])
                for i, row in enumerate(matrix):
                    for aa, value in zip(amino_acids, row):
                        writer.writerow([i + 36, sequence[i], aa, float(value)])
            positions = np.arange(len(sequence)) + 36
            selected = np.flatnonzero(positions < 280)
            ranking = sorted(selected, key=lambda i: (-float(matrix[i].mean()), int(i)))[:10]
            top10 = [
                {"position": int(positions[i]), "wild_type": sequence[i], "score": float(matrix[i].mean())}
                for i in ranking
            ]
            report["comparisons"].append(
                {
                    "checkpoint": weight.name,
                    "indexing": label,
                    "mask_token_offset": offset,
                    "top10": top10,
                    "matrix_sha256": sha(args.output / (prefix + ".npy")),
                }
            )
            if offset == 0:
                # Execute only author function definitions, never notebook plotting,
                # top-level imports, downloads or training commands.
                functions = [n for n in ast.parse(notebook_source).body if isinstance(n, ast.FunctionDef)]
                namespace = {"torch": torch, "vocab": vocab, "model": model, "device": device}
                exec(
                    compile(ast.Module(body=functions, type_ignores=[]), "author-notebook-functions", "exec"), namespace
                )
                reference = np.asarray(namespace["check_heatmap"]())
                error = float(np.max(np.abs(reference - matrix)))
                if not np.allclose(reference, matrix, rtol=1e-6, atol=1e-6):
                    raise ValueError("author replay differs")
                report["comparisons"][-1]["independent_author_function_max_abs_error"] = error
            json_save(args.output / "report.json", report)
            print(
                json.dumps(
                    {"checkpoint": weight.name, "indexing": label, "top10_positions": [x["position"] for x in top10]}
                ),
                flush=True,
            )
        del model
    measurements = list(csv.DictReader((source / "exp_data.txt").open(), delimiter="\t"))
    grouped = {}
    normalized = []
    for index, row in enumerate(measurements, 2):
        mutation = row["mutation"]
        pos = int(mutation[1:-1])
        wild = mutation[0]
        if not 0 <= pos - 36 < len(sequence) or sequence[pos - 36] != wild:
            raise ValueError("experimental mutation reference mismatch")
        record = {"source_line": index, "mutation": mutation, "Tm": float(row["Tm"]), "U_per_mg": float(row["U/mg"])}
        normalized.append(record)
        grouped.setdefault(mutation, []).append(record)
    json_save(args.output / "measurements_preserved.json", normalized)
    report.update(
        status="completed",
        measurement_rows=len(normalized),
        duplicate_mutations={k: v for k, v in grouped.items() if len(v) > 1},
        duplicate_policy="preserve both source rows; no label averaging or correlation claim",
        wall_seconds=round(time.monotonic() - started, 3),
        missing_for_complete_paper=[
            "original training MSA and exact training environment",
            "full experimental processing and conditions",
            "wet-lab PET depolymerization",
        ],
    )
    json_save(args.output / "report.json", report)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "research_runs/rsi/case_assets/code-for-TurboPETase" / COMMIT
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["PF01083", "PTM311", "pretrained-model"],
        default=["PF01083", "PTM311", "pretrained-model"],
    )
    parser.add_argument("--device", default="cuda:0")
    raise SystemExit(main(parser.parse_args()))
