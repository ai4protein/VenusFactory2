"""Administrator-configured subprocess runtimes for built-in mutation models.

No model is imported into the API process. Runtime profiles choose environments
and checkpoints; users can only select a known model, never a script or command.
These are trusted scientific workers, not the generated-code sandbox.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from mcp_tools.models.execution import RuntimeCommand, environment, execute
from pathlib import Path

from mcp_tools.models.profiles import PROJECT, load_runtime_settings, repository_path

SPECS = {
    "VenusREM": ("prosst", "venusrem", "structure", "AI4Protein/ProSST-2048"),
    "ProSST": ("prosst", "prosst", "structure", "AI4Protein/ProSST-2048"),
    "ESM-1v": ("esm1v", "esm1v", "sequence", "facebook/esm1v_t33_650M_UR90S_1"),
    "ESM-IF1": ("esmif1", "esmif1", "structure", "esm_if1_gvp4_t16_142M_UR50"),
    "ProtSSN": ("protssn", "protssn", "structure", "facebook/esm2_t33_650M_UR50D"),
}
PROSST_VOCABULARIES = (20, 128, 512, 1024, 2048, 4096)
SPECS.update({f"ProSST-{v}": ("prosst", "prosst", "structure", f"AI4Protein/ProSST-{v}") for v in PROSST_VOCABULARIES})


def prosst_settings(model, settings):
    """A variant must never silently reuse the default 2048 checkpoint."""
    size = 2048 if model == "ProSST" else int(model.split("-")[-1])
    cfg = dict(settings.get("prosst", {}))
    if size != 2048:
        cfg.pop("weights", None)
        cfg.pop("model_id", None)
    cfg.update(cfg.get("variants", {}).get(str(size), {}))
    return size, cfg


ALIASES = {"VenusREM (foldseek-based)": "VenusREM", "ProSST-2048": "ProSST", "protssn": "ProtSSN", "ESM-IF": "ESM-IF1", "ESM-1V": "ESM-1v", "prosst": "ProSST", "esm1v": "ESM-1v", "esmif1": "ESM-IF1"}


def canonical_model(name):
    return ALIASES.get(name, name)


def build_command(model, model_type, source, output, *, mutations=None, settings=None):
    model = canonical_model(model)
    if model not in SPECS:
        raise ValueError("No mutation runtime for this model")
    key, script, expected, default_id = SPECS[model]
    if model_type != expected:
        raise ValueError(f"{model} requires a {expected} input")
    settings = load_runtime_settings() if settings is None else settings
    cfg = dict(settings.get(key, {}))
    if key == "prosst":
        vocab_size, cfg = prosst_settings("ProSST" if model == "VenusREM" else model, settings)
    if cfg.get("enabled") is False:
        raise ValueError(f"{model} runtime is disabled")
    python = repository_path(cfg["python"]) if cfg.get("python") else Path(sys.executable)
    if not python.is_file():
        raise ValueError(f"{model} runtime Python is not installed")
    source = Path(source).absolute()
    if not source.is_file():
        raise ValueError("Prediction input does not exist")
    output = Path(output).absolute()
    weights = cfg.get("weights")
    if weights:
        weights = repository_path(weights)
        if not weights.exists():
            raise ValueError(f"{model} runtime weights are not installed")
    model_id = str(weights or cfg.get("model_id") or default_id)
    if key == "prosst":
        from mcp_tools.models.checkpoint_cache import cached_model
        cache = str(repository_path(cfg['cache']) / 'huggingface/hub') if cfg.get('cache') else None
        model_id = cached_model(model_id, cache)
    argv = [str(python), str(PROJECT / "mcp_tools/models/pretrained/mutation/models" / (script + ".py")),
            "--pdb_file" if expected == "structure" else "--fasta_file", str(source),
            "--output_csv", str(output)]
    if model == "ProtSSN":
        argv += ["--plm_model", str(cfg.get("model_id") or default_id)]
        if weights: argv += ["--gnn_model_path", str(weights)]
        if cfg.get("config"): argv += ["--gnn_config", str(repository_path(cfg["config"]))]
        if cfg.get("use_ensemble") is False: argv += ["--single_model"]
    elif model != "VenusREM":
        argv += ["--model_name", model_id]
    if mutations:
        argv += ["--mutations_csv", str(Path(mutations).absolute())]
    if key == "prosst" and model != "VenusREM": argv += ["--structure_vocab_size", str(vocab_size)]
    if key == "prosst" and cfg.get("structure_weights"):
        structure = repository_path(cfg["structure_weights"])
        if not (structure / "AE.pt").is_file() or not (structure / f"{vocab_size}.joblib").is_file():
            raise ValueError("ProSST structure tokenizer weights are incomplete")
        if model != "VenusREM": argv += ["--structure_weights", str(structure)]
    env = environment(cfg)
    timeout = max(1, min(int(cfg.get("timeout_seconds", 3600)), 86400))
    return RuntimeCommand(argv, env, output.parent, timeout, key)


def score_mutations(model, source, mutants):
    """Ensemble callers use the same runtime and retain the input mutation order."""
    import pandas as pd
    model = canonical_model(model)
    if model not in SPECS:
        raise ValueError("Unknown mutation model")
    with tempfile.TemporaryDirectory(prefix="venus-mutation-") as folder:
        folder = Path(folder)
        mutations = folder / "mutations.csv"
        pd.DataFrame({"mutant": mutants}).to_csv(mutations, index=False)
        output = folder / "scores.csv"
        execute(build_command(model, SPECS[model][2], source, output, mutations=mutations))
        df = pd.read_csv(output)
        column = SPECS[model][1] + "_score"
        values = df.drop_duplicates("mutant").set_index("mutant")[column]
        return [float(values[m]) for m in mutants]
