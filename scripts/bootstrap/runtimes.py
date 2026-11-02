from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

from .assets import clone, download, extract, github_asset
from .common import add_missing, atomic_json, digest, inside, read_json


HF_SCRIPT = """
import json, sys
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
repo, target, pattern = sys.argv[1:]
root = Path(target)
marker = root / '.venus-snapshot.json'
if marker.exists():
    info = json.loads(marker.read_text())
    revision = info['revision']
else:
    revision = HfApi(token=False).model_info(repo).sha
snapshot_download(repo, revision=revision, local_dir=root, token=False,
                  allow_patterns=pattern.split(',') if pattern else None)
marker.write_text(json.dumps({'repository': repo, 'revision': revision}))
"""


def snapshot(run, python, repo, folder, pattern=""):
    run([python, "-c", HF_SCRIPT, repo, folder, pattern])


class Runtimes:
    def __init__(self, root, env, python, manifest, gpu=False):
        self.root, self.env, self.python, self.manifest = root, env, python, manifest
        self.run = env.run
        self.base = inside(root, "runtime/local/model-runtimes")
        self.config = root / "configs/model_runtimes.json"
        self.gpu = gpu

    def merge(self, name, config):
        original = read_json(self.config, {})
        merged = add_missing(original, {name: config})
        if merged != original:
            if self.config.exists():
                backup = self.root / "runtime/local/bootstrap/backups" / f"model_runtimes-{time.time_ns()}.json"
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.config, backup)
            atomic_json(self.config, merged)

    def relative(self, path):
        path = Path(path)
        return str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)

    def install(self, name):
        spec = self.manifest["tools"][name]
        if spec.get("manual"):
            from .checks import tool_health
            health = tool_health(self.root, name, self.python)
            if health["status"] == "installed":
                return {"status": "existing", "message": "Existing manually provisioned runtime passed installation checks", "health": health}
            return {"status": "manual", "message": spec["manual"]}
        if spec.get("gpu") and not self.gpu:
            return {"status": "blocked", "message": "This optional runtime requires an NVIDIA GPU"}
        target = inside(self.root, self.base / name)
        receipt_path = target / "installation.json"
        receipt = read_json(receipt_path, {})
        signature = hashlib.sha256((digest(self.root / "configs/bootstrap.json") + digest(Path(__file__))
                                    + digest(Path(__file__).with_name("assets.py"))
                                    + digest(Path(__file__).with_name("system.py"))).encode()).hexdigest()
        if receipt.get("recipe_sha256") == signature and receipt.get("verification"):
            required = [self.root / p for p in receipt.get("required_files", [])]
            valid = required and all(p.is_file() and p.stat().st_size for p in required)
            if valid:
                valid = all(digest(self.root / path) == info["sha256"] for path, info in receipt.get("assets", {}).items())
            if valid:
                self.merge(name, receipt["profile"])
                return {"status": "ready" if receipt["profile"].get("enabled") else "needs_validation",
                        "verification": receipt["verification"], "message": "Previously checked installation reused"}
        existing = read_json(self.config, {}).get(name, {})
        # Administrator-owned profiles may point outside bootstrap-managed roots.
        # Never replace them or silently switch a live service to new weights.
        if existing and not receipt:
            example = read_json(self.root / "configs/model_runtimes.example.json", {}).get(name, {})
            if existing != example:
                return {"status": "existing", "message": "Administrator profile preserved; use doctor to audit it"}
            # Untouched, disabled templates contain no administrator installation.
            # Preserve their switch and limits, replace only uninstalled example paths.
            original = read_json(self.config, {})
            old_paths = {k for k in existing if k in {"python", "root", "weights", "packing_weights", "structure_weights", "executable", "installation", "cache", "model_dir", "confidence_model_dir", "config"}}
            if any((self.root / existing[k]).exists() for k in old_paths):
                return {"status": "existing", "message": "Existing template installation preserved"}
            # Record the pending replacement; apply it only after assets and smoke pass.
            template_paths = old_paths
        else:
            template_paths = set()
        target.mkdir(parents=True, exist_ok=True)
        source = target / "source"
        revision = None
        if spec.get("repository"):
            revision = clone(self.run, spec["repository"], spec["revision"], source)
        python = self.python
        if spec.get("python"):
            python = self.env.create(target / "env", spec["python"])
            marker = target / "dependencies.json"
            dependency_key = {"recipe": signature, "torch": self.env.torch_type}
            if read_json(marker) != dependency_key:
                if name == "diffdock":
                    self.diffdock_dependencies(python)
                elif name == "boltzgen":
                    self.run([python, "-m", "pip", "install", "torch==2.7.1", "--index-url", "https://download.pytorch.org/whl/cu126"])
                    self.env.pip(python, *spec["packages"])
                else:
                    self.env.pip(python, "torch==2.8.0", torch=True)
                    self.env.pip(python, *spec["packages"])
                atomic_json(marker, dependency_key)
        profile = {"enabled": False, "python": self.relative(python),
                   "installation": self.relative(receipt_path), "timeout_seconds": 1800}
        required = []
        if name == "proteinmpnn":
            for variant in ["vanilla", "soluble", "ca"]:
                weights = source / f"{variant}_model_weights"
                if not list(weights.glob("*.pt")):
                    raise ValueError(f"Upstream omitted {variant} ProteinMPNN weights")
                required += list(weights.glob("*.pt"))
            profile["weights"] = self.relative(source)
            profile["root"] = self.relative(source)
        elif name == "ligandmpnn":
            weights = source / "model_params/ligandmpnn_v_32_010_25.pt"
            download("https://files.ipd.uw.edu/pub/ligandmpnn/" + weights.name, weights,
                     expected="161cd264061fda9680cbb940255522ae42f2966c552d045d87913d9452a80970")
            packing = source / "model_params/ligandmpnn_sc_v_32_002_16.pt"
            download("https://files.ipd.uw.edu/pub/ligandmpnn/" + packing.name, packing)
            required += [weights, packing]
            profile.update(root=self.relative(source), weights=self.relative(weights), packing_weights=self.relative(packing))
        elif name == "antibmpnn":
            archive = download("https://zenodo.org/records/13387792/files/model_weights.zip", target / "model_weights.zip", max_bytes=512 * 2**20)
            import zipfile
            weights = source / "antibmpnn_model_weights/antibmpnn_000.pt"
            weights.parent.mkdir(exist_ok=True)
            if not weights.is_file():
                with zipfile.ZipFile(archive) as z:
                    matches = [i for i in z.infolist() if Path(i.filename).name == weights.name]
                    if len(matches) != 1 or matches[0].file_size > 512 * 2**20:
                        raise ValueError("Unexpected AntiBMPNN weight archive")
                    with z.open(matches[0]) as inp, weights.open("wb") as out:
                        shutil.copyfileobj(inp, out)
            script = source / "Running_AntiBMPNN_run.py"
            self.patch(script, "path_to_model_weights='../antibmpnn_model_weights'", "path_to_model_weights=args.path_to_model_weights")
            self.patch(script, "seed=int(np.random.randint(0, high=999, size=1, dtype=int)[0])", "seed=int(args.seed or 1)")
            required += [weights]
            profile.update(root=self.relative(source), weights=self.relative(weights.parent))
        elif name == "s4pred":
            archive = download("https://bioinf.cs.ucl.ac.uk/downloads/s4pred/weights.tar.gz", target / "weights.tar.gz",
                               expected="e04ad7d10b61551f7e07a86b65bb88dc", algorithm="md5")
            required += [source / f"weights/weights_{i}.pt" for i in range(1, 6)]
            if not all(p.is_file() for p in required):
                extract(archive, source)
            profile["root"] = self.relative(source)
        elif name in {"prosst", "esm1v", "protssn", "esmif1"}:
            required += self.foundation(name, profile, source, target)
        elif name == "vina":
            import platform
            if platform.machine() != "x86_64":
                raise ValueError("Vina binary recipe currently requires Linux x86_64")
            binary = github_asset("forlilab/AutoDock-Vina", spec["release"], "vina_1.2.7_linux_x86_64", target / "bin/vina")
            binary.chmod(0o755)
            profile = {"enabled": False, "executable": self.relative(binary), "cpus": 2, "installation": self.relative(receipt_path)}
            required += [binary]
        elif name == "admet_ai":
            # The package ships/downloads the ensemble through its official API.
            self.run([python, "-c", "from admet_ai import ADMETModel; m=ADMETModel(); p=m.predict(smiles=['CCO']); assert len(p)==1"])
            required += [python]
        elif name == "boltzgen":
            binary = target / "env/bin/boltzgen"
            cache = target / "cache"
            self.run([binary, "download", "all", "--cache", cache], timeout=14400)
            profile.update(executable=self.relative(binary), cache=self.relative(cache), timeout_seconds=3600, cpus=2)
            required += [binary, *cache.rglob("*.ckpt")]
        elif name == "diffdock":
            archive = download("https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip", target / "models.zip")
            if not (source / "workdir/v1.1/score_model").is_dir():
                extract(archive, source / "workdir/v1.1")
            self.patch(source / "inference.py", "(64000, rlimit[1])", "(min(64000, rlimit[1]), rlimit[1])")
            cache = target / "cache"
            self.run([python, "-c", "import esm; esm.pretrained.esm2_t33_650M_UR50D()"], extra_env={"TORCH_HOME": str(cache)})
            profile.update(root=self.relative(source), model_dir=self.relative(source / "workdir/v1.1/score_model"),
                           confidence_model_dir=self.relative(source / "workdir/v1.1/confidence_model"),
                           config=self.relative(source / "default_inference_args.yaml"), confidence_ckpt="best_model_epoch75.pt",
                           cache=self.relative(cache), cpus=2, timeout_seconds=3600)
            required += list((source / "workdir/v1.1").rglob("*.pt"))
        if not required or not all(p.is_file() and p.stat().st_size for p in required):
            raise ValueError(f"Incomplete assets for {name}")
        verification = self.smoke(name, python, source, target, profile)
        profile["enabled"] = verification == "inference"
        if name in {"prosst", "esm1v", "esmif1", "protssn"}:
            profile["enabled"] = True  # The model-loading check is explicit in the receipt.
        receipt = {"runtime": name, "verified": verification == "inference", "verification": verification,
                   "recipe_sha256": signature, "revision": revision, "repository": spec.get("repository"),
                   "required_files": [self.relative(p) for p in required], "profile": profile,
                   "assets": {self.relative(p): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in required},
                   "installed_at": time.time()}
        atomic_json(receipt_path, receipt)
        if template_paths:
            original = read_json(self.config, {})
            if original.get(name) != existing:
                raise ValueError("Runtime configuration changed during installation; review it before merging")
            backup = self.root / "runtime/local/bootstrap/backups" / f"model_runtimes-{time.time_ns()}.json"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.config, backup)
            original[name] = {k: v for k, v in existing.items() if k not in template_paths}
            original[name] = add_missing(original[name], profile)
            atomic_json(self.config, original)
        else:
            self.merge(name, profile)
        return {"status": "ready" if profile["enabled"] else "needs_validation", "verification": verification}

    @staticmethod
    def patch(path, old, new):
        text = path.read_text()
        if new in text:
            return
        if text.count(old) != 1:
            raise ValueError(f"Upstream compatibility patch does not match: {path.name}")
        backup = path.with_suffix(path.suffix + ".upstream")
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(text.replace(old, new), encoding="utf-8")

    def foundation(self, name, profile, source, target):
        if name == "esmif1":
            weights = download("https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt", target / "weights/esm_if1_gvp4_t16_142M_UR50.pt")
            profile.update(weights=self.relative(weights))
            return [weights]
        model = {"prosst": "AI4Protein/ProSST-2048", "esm1v": "facebook/esm1v_t33_650M_UR90S_1", "protssn": "facebook/esm2_t33_650M_UR50D"}[name]
        weights = target / "weights"
        if name == "protssn":
            cache = target / "cache/huggingface/hub"
            record = target / "foundation.json"
            code = """import sys,json
from pathlib import Path
from huggingface_hub import snapshot_download
model,cache,record=sys.argv[1:]
marker=Path(record)
revision=json.loads(marker.read_text())['revision'] if marker.exists() else 'main'
folder=snapshot_download(model,revision=revision,cache_dir=cache,token=False)
marker.write_text(json.dumps({'revision':Path(folder).name,'snapshot':folder}))
"""
            self.run([self.python, "-c", code, model, cache, record])
            weights = Path(read_json(record)["snapshot"])
            profile["cache"] = self.relative(target / "cache")
        else:
            snapshot(self.run, self.python, model, weights)
        profile.update(weights=self.relative(weights), model_id=model)
        required = list(weights.glob("*.safetensors")) or list(weights.glob("*.bin"))
        if name == "prosst":
            structure = target / "structure-tokenizer/weights"
            structure.mkdir(parents=True, exist_ok=True)
            manifest = read_json(self.root / "mcp_tools/models/pretrained/prosst/weights-manifest.json", {})
            for filename, info in manifest.items():
                src = source / "prosst/structure/static" / filename
                if not src.is_file():
                    raise ValueError(f"ProSST upstream omitted tokenizer asset: {filename}")
                expected = info.get("sha256")
                if expected and digest(src) != expected:
                    raise ValueError("ProSST tokenizer differs from the supported version")
                dest = structure / filename
                if not dest.exists():
                    shutil.copy2(src, dest)
                required.append(dest)
            profile["structure_weights"] = self.relative(structure)
        elif name == "protssn":
            archive_root = target / "archive"
            snapshot(self.run, self.python, "tyang816/ProtSSN", archive_root, "ProtSSN.zip")
            unpack = target / "unpacked"
            extract(archive_root / "ProtSSN.zip", unpack)
            gnn = target / "gnn"
            gnn.mkdir(exist_ok=True)
            for k in (10, 20, 30):
                for h in (512, 768, 1280):
                    filename = f"protssn_k{k}_h{h}.pt"
                    matches = list(unpack.rglob(filename))
                    if len(matches) != 1:
                        raise ValueError("Missing or duplicate ProtSSN checkpoint")
                    shutil.copy2(matches[0], gnn / filename)
                    required.append(gnn / filename)
            profile.update(weights=self.relative(gnn), model_id=model, use_ensemble=True)
        return required

    def diffdock_dependencies(self, python):
        self.run([python, "-m", "pip", "install", "torch==2.2.2", "--index-url", "https://download.pytorch.org/whl/cu121"])
        self.env.pip(python, "numpy<2", "scipy==1.12.0", "e3nn==0.5.1", "fair-esm==2.0.0", "networkx==2.8.4", "pandas==1.5.3",
                     "rdkit", "scikit-learn", "biopython", "pyyaml", "requests", "tqdm", "torch-geometric==2.5.3",
                     "torchmetrics==0.11.0", "pytorch-lightning==1.9.5", "spyrmsd", "prody")
        self.env.pip(python, "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv",
                     links="https://data.pyg.org/whl/torch-2.2.2+cu121.html")

    def smoke(self, name, python, source, target, profile):
        output = target / "smoke"
        output.mkdir(exist_ok=True)
        cpu = {"CUDA_VISIBLE_DEVICES": ""}
        if name in {"proteinmpnn", "ligandmpnn", "antibmpnn"}:
            pdb = download("https://files.rcsb.org/download/1CRN.pdb", target / "smoke/1CRN.pdb", max_bytes=2**20)
            if name == "ligandmpnn":
                args = [source / "run.py", "--model_type", "ligand_mpnn", "--pdb_path", pdb,
                        "--checkpoint_ligand_mpnn", self.root / profile["weights"], "--out_folder", output,
                        "--number_of_batches", "1", "--batch_size", "1", "--seed", "193"]
            else:
                script = "protein_mpnn_run.py" if name == "proteinmpnn" else "Running_AntiBMPNN_run.py"
                weights = source / "vanilla_model_weights" if name == "proteinmpnn" else self.root / profile["weights"]
                args = [source / script, "--pdb_path", pdb, "--out_folder", output,
                        "--path_to_model_weights", weights, "--num_seq_per_target", "1", "--batch_size", "1", "--seed", "193"]
                if name == "antibmpnn":
                    args += ["--model_name", "antibmpnn_000"]
            self.run([python, *args], cwd=source, timeout=300, extra_env=cpu)
            files = list((output / "seqs").glob("*.fa*"))
            if not files or not any(p.read_text().count(">") >= 2 for p in files):
                raise ValueError("MPNN smoke test produced no designed sequence")
            return "inference"
        if name == "s4pred":
            fasta = output / "input.fasta"
            fasta.write_text(">ubiquitin\nMQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG\n")
            self.run([python, source / "run_model.py", fasta, "--device", "cpu", "--threads", "2",
                      "--save-files", "--save-by-idx", "--outdir", output], timeout=300)
            ss2 = output / "s4_out_0.ss2"
            if not ss2.is_file() or sum(bool(s.split()) and s.split()[0].isdigit() for s in ss2.read_text().splitlines()) != 76:
                raise ValueError("S4PRED smoke test did not predict all residues")
            return "inference"
        if name == "admet_ai":
            return "inference"
        if name == "vina":
            self.run([self.root / profile["executable"], "--version"], timeout=30)
            return "cli"
        if name in {"boltzgen", "diffdock"}:
            command = [self.root / profile["executable"], "run", "--help"] if name == "boltzgen" else [python, source / "inference.py", "--help"]
            self.run(command, cwd=source if source.exists() else target, timeout=1200)
            return "cli"
        if name == "esmif1":
            code = "import sys; from mcp_tools.models.pretrained.mutation.models.esm import pretrained; pretrained.load_model_and_alphabet(sys.argv[1])"
            self.run([python, "-c", code, self.root / profile["weights"]], timeout=600, extra_env=cpu)
        else:
            path = Path(read_json(target / "foundation.json")["snapshot"]) if name == "protssn" else target / "weights"
            code = "import sys; from transformers import AutoModelForMaskedLM,AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=True); AutoModelForMaskedLM.from_pretrained(sys.argv[1],local_files_only=True,trust_remote_code=True)"
            self.run([python, "-c", code, path], timeout=600, extra_env=cpu)
        return "model_load"
