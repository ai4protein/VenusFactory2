"""ESMFold v1 runs in its configured interpreter, never in the API process."""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path
from mcp_tools.models.execution import command, execute, profile
from mcp_tools.models.profiles import repository_path

def model_path(settings=None):
    cfg, _ = profile("esmfold", settings)
    return str(repository_path(cfg["weights"])) if cfg.get("weights") else "facebook/esmfold_v1"

def run(sequence, output_dir, output_file=None, *, settings=None):
    cfg, _ = profile("esmfold", settings)
    sequence = "".join(str(sequence).split()).upper()
    if not sequence or any(c not in "ACDEFGHIKLMNPQRSTVWYX" for c in sequence):
        raise ValueError("Enter a protein amino acid sequence")
    if len(sequence) > int(cfg.get("max_sequence_length", 2048)):
        raise ValueError("Sequence exceeds this ESMFold runtime limit")
    folder = Path(output_dir).absolute()
    folder.mkdir(parents=True, exist_ok=True)
    destination = Path(output_file).absolute() if output_file else folder / "structure.pdb"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".esmfold-", dir=folder) as tmp:
        request = Path(tmp) / "request.json"
        result = Path(tmp) / "result.json"
        request.write_text(json.dumps({"sequence": sequence, "output_file": str(destination),
                                      "result_file": str(result), "weights": model_path(settings),
                                      "chunk_size": int(cfg.get("fold_chunk_size", 64))}))
        request.chmod(0o600)
        execute(command("esmfold", ["-m", __name__, str(request)], settings=settings))
        info = json.loads(result.read_text())
        if not destination.is_file() or not destination.stat().st_size:
            raise RuntimeError("ESMFold produced no structure")
        return str(destination), info

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    values = json.loads(args.request.read_text())
    from mcp_tools.models.pretrained.structure.esmfold import _predict_local_in_process
    path, info = _predict_local_in_process(
        values["sequence"], str(Path(values["output_file"]).parent), values["output_file"],
        model_name=values["weights"], chunk_size=values["chunk_size"])
    if not path or not info:
        raise RuntimeError("ESMFold inference failed")
    Path(values["result_file"]).write_text(json.dumps(info))

if __name__ == "__main__":
    main()
