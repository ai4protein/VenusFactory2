"""CataPro source-bound features and independent checkpoint arithmetic.

These helpers reproduce released computation. They do not establish a new model
or unbiased generalization: the author's CV folds also select checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import sqlite3
from pathlib import Path

import numpy as np

DATASETS = {
    "kcat": "kcat-data_0.4simi-10fold.csv",
    "km": "Km-data_0.4simi-10fold.csv",
    "activity": "kcat-over-Km-data_0.4simi-10fold.csv",
}
COLUMNS = [
    "pred_log10[kcat(s^-1)]",
    "pred_log10[Km(mM)]",
    "pred_log10[kcat/Km(s^-1mM^-1)]",
]


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def checked_manifest(root, name):
    manifest = json.loads((root / name).read_text())
    if manifest["status"] != "admitted":
        raise ValueError("Asset admission incomplete")
    for row in manifest["files"]:
        if file_sha(root / row["path"]) != row["sha256"]:
            raise ValueError("Admitted source changed")
    return manifest


def normalize_protein(sequence):
    if not sequence:
        raise ValueError("Empty protein")
    if len(sequence) > 1000:
        sequence = sequence[:500] + sequence[-500:]
    return re.sub(r"[UZOB]", "X", sequence)


def log_targets(row):
    """CSV stores M; released prediction headers specify mM."""
    result = {}
    for field in ["kcat(s^-1)", "Km(M)"]:
        if field in row:
            value = float(row[field])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Kinetic targets must be positive and finite")
            result["kcat" if field == "kcat(s^-1)" else "km"] = math.log10(value) + (3.0 if field == "Km(M)" else 0.0)
    if "kcat" in result and "km" in result:
        result["activity"] = result["kcat"] - result["km"]
    return result


def dataset_rows(source, name):
    path = source / ("samples/sample_inp.csv" if name == "sample" else "datasets/" + DATASETS[name])
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for i, row in enumerate(rows):
        if name == "sample":
            item = {
                "id": row["Enzyme_id"] + "_" + row["type"],
                "sequence": row["sequence"],
                "smiles": row["smiles"],
                "fold": None,
            }
        else:
            fold = float(row["fold"])
            if not fold.is_integer() or not 0 <= fold < 10:
                raise ValueError("Invalid author fold")
            item = {
                "id": row[""],
                "sequence": row["Sequence"],
                "smiles": row["Smiles"],
                "fold": int(fold),
                "targets": log_targets(row),
            }
        if not item["sequence"] or not item["smiles"]:
            raise ValueError("Empty input")
        result.append({**item, "row": i})
    return result


def encoder_binding(path):
    names = ["config.json", "pytorch_model.bin", "spiece.model", "special_tokens_map.json", "tokenizer_config.json"]
    return {"path": str(path.resolve()), "files": {n: file_sha(path / n) for n in names}}


class FeatureCache:
    """Content-bound unsupervised features; no labels or learned task parameters."""

    def __init__(self, path, contract):
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (contract TEXT NOT NULL)")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, dimension INTEGER, data BLOB, sha256 TEXT)"
        )
        encoded = json.dumps(contract, sort_keys=True)
        prior = self.db.execute("SELECT contract FROM metadata").fetchall()
        if prior and (len(prior) != 1 or prior[0][0] != encoded):
            self.db.close()
            raise ValueError("Feature cache contract differs")
        if not prior:
            self.db.execute("INSERT INTO metadata VALUES (?)", (encoded,))
        self.db.commit()

    @staticmethod
    def key(kind, value):
        return kind + ":" + hashlib.sha256(value.encode()).hexdigest()

    def get(self, kind, value, dimension):
        row = self.db.execute(
            "SELECT dimension,data,sha256 FROM vectors WHERE key=?", (self.key(kind, value),)
        ).fetchone()
        if row is None:
            return None
        if row[0] != dimension or hashlib.sha256(row[1]).hexdigest() != row[2]:
            raise ValueError("Cached vector integrity failed")
        array = np.frombuffer(row[1], dtype="<f4").copy()
        if array.shape != (dimension,) or not np.isfinite(array).all():
            raise ValueError("Invalid cached feature")
        return array

    def put(self, kind, value, array):
        array = np.asarray(array, dtype="<f4")
        if array.ndim != 1 or not np.isfinite(array).all():
            raise ValueError("Invalid feature vector")
        data = array.tobytes()
        self.db.execute(
            "INSERT INTO vectors VALUES (?,?,?,?)",
            (self.key(kind, value), len(array), data, hashlib.sha256(data).hexdigest()),
        )
        self.db.commit()

    def close(self):
        self.db.close()


class FeatureBuilder:
    def __init__(self, protein_path, molecule_path, cache, device="cuda:0"):
        from transformers import T5EncoderModel, T5Tokenizer

        self.cache = cache
        self.device = device
        self.protein_tokenizer = T5Tokenizer.from_pretrained(protein_path, do_lower_case=False, local_files_only=True)
        self.protein = T5EncoderModel.from_pretrained(protein_path, local_files_only=True).to(device).eval()
        self.molecule_tokenizer = T5Tokenizer.from_pretrained(molecule_path, local_files_only=True)
        # The released molecular encoder runs on CPU.
        self.molecule = T5EncoderModel.from_pretrained(molecule_path, local_files_only=True).eval()

    def protein_vector(self, sequence):
        import torch

        cached = self.cache.get("protein", sequence, 1024)
        if cached is not None:
            return cached
        tokens = self.protein_tokenizer.batch_encode_plus(
            [" ".join(normalize_protein(sequence))], add_special_tokens=True, padding=True
        )
        ids = torch.tensor(tokens["input_ids"]).to(self.device)
        mask = torch.tensor(tokens["attention_mask"]).to(self.device)
        with torch.no_grad():
            hidden = self.protein(input_ids=ids, attention_mask=mask).last_hidden_state[0]
        residues = hidden[: int(mask.sum()) - 1].cpu().numpy()
        # Float64 accumulation follows the released Python summation, then the
        # original inference boundary casts the concatenated tensor to float32.
        vector = np.mean(residues, axis=0, dtype=np.float64).astype(np.float32)
        self.cache.put("protein", sequence, vector)
        return vector

    def molecule_vector(self, smiles):
        import torch

        cached = self.cache.get("molecule", smiles, 768)
        if cached is not None:
            return cached
        ids = self.molecule_tokenizer(smiles, return_tensors="pt").input_ids
        with torch.no_grad():
            hidden = self.molecule(input_ids=ids).last_hidden_state
            vector = torch.mean(hidden[0][:-1, :], dim=0).cpu().numpy()
        self.cache.put("molecule", smiles, vector)
        return vector

    def fingerprint(self, smiles):
        from rdkit import Chem
        from rdkit.Chem import MACCSkeys

        cached = self.cache.get("maccs", smiles, 167)
        if cached is not None:
            return cached
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise ValueError("Invalid substrate SMILES")
        vector = np.array([int(x) for x in MACCSkeys.GenMACCSKeys(molecule).ToBitString()], dtype=np.float32)
        self.cache.put("maccs", smiles, vector)
        return vector

    def row(self, row):
        return np.concatenate(
            [self.protein_vector(row["sequence"]), self.molecule_vector(row["smiles"]), self.fingerprint(row["smiles"])]
        )


def author_models(source):
    modules = {}
    for name in ["model", "act_model"]:
        spec = importlib.util.spec_from_file_location("catapro_verified_" + name, source / "inference" / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def _linear(x, state, prefix):
    return x @ state[prefix + ".weight"].T + state[prefix + ".bias"]


def _norm(x, state, prefix):
    return (x - state[prefix + ".running_mean"]) / np.sqrt(state[prefix + ".running_var"] + 1e-5) * state[
        prefix + ".weight"
    ] + state[prefix + ".bias"]


def _hidden(features, state, prefix=""):
    x = np.concatenate(
        [
            _norm(features[:, :1024], state, prefix + "prot_norm"),
            _norm(features[:, 1024:1792], state, prefix + "molt5_norm"),
            features[:, 1792:],
        ],
        axis=1,
    )
    return np.maximum(_norm(_linear(x, state, prefix + "decoder.0"), state, prefix + "decoder.1"), 0)


def numpy_forward(features, state, activity=False):
    """Independent NumPy replay of released eval-mode PyTorch arithmetic."""
    state = {k: v.detach().cpu().numpy() if hasattr(v, "detach") else v for k, v in state.items()}
    h = _hidden(features, state)
    if not activity:
        return _linear(h, state, "out.0")[:, 0]
    kcat = _linear(_hidden(features, state, "kcat_model."), state, "kcat_model.out.0")
    km = _linear(_hidden(features, state, "Km_model."), state, "Km_model.out.0")
    attention = _linear(h, state, "attn.0")
    attention = np.exp(attention - attention.max(axis=1, keepdims=True))
    attention /= attention.sum(axis=1, keepdims=True)
    correction = _linear(attention * h, state, "out")
    return (0.6 * (kcat - km) + 0.4 * correction)[:, 0]
