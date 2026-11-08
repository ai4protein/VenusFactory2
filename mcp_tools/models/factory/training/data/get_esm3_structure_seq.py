"""ESM3 structure tokens, dispatched to an independently installed runtime."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

VQVAE_CODEBOOK_SIZE = 4096
VQVAE_SPECIAL_TOKENS = dict(zip(
    ("MASK", "EOS", "BOS", "PAD", "CHAINBREAK"), range(4096, 4101)
))


def ESM3_structure_encoder_v0(device="cpu", weights=None):
    """Low-level runtime API; Factory orchestration uses encode_structures."""
    import torch
    from esm.models.vqvae import StructureTokenEncoder

    if weights is None:
        from mcp_tools.models.profiles import load_runtime_settings
        weights = load_runtime_settings().get("esm3", {}).get("weights")
    if not weights or not Path(weights).is_file():
        raise ValueError("An installed ESM3 structure-encoder checkpoint is required")
    model = StructureTokenEncoder(
        d_model=1024, n_heads=1, v_heads=128, n_layers=2, d_out=128, n_codes=4096,
    ).to(device).eval()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model


def get_esm3_structure_seq(pdb_file, encoder, device="cpu"):
    import numpy as np
    import torch
    import biotite.structure.io as bsio
    from biotite.structure.io.pdb import PDBFile
    from esm.utils.structure.protein_chain import ProteinChain

    pdb_file = str(pdb_file)
    chain_ids = np.unique(PDBFile.read(pdb_file).get_structure().chain_id)
    chain = ProteinChain.from_pdb(pdb_file, chain_id=chain_ids[0])
    coords, _, residue_index = chain.to_structure_encoder_inputs()
    structure = bsio.load_structure(pdb_file, extra_fields=["b_factor"])
    with torch.inference_mode():
        _, tokens = encoder.encode(coords.to(device), residue_index=residue_index.to(device))
    return {
        "name": Path(pdb_file).name.split(".")[0],
        "esm3_structure_seq": tokens.cpu().tolist()[0],
        "plddt": float(structure.b_factor.mean()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--pdb_file")
    inputs.add_argument("--pdb_dir")
    inputs.add_argument("--pdb_manifest", help=argparse.SUPPRESS)
    parser.add_argument("--out_file", default="esm3_structure_seq.csv")
    parser.add_argument("--out_format", choices=("csv", "json"), default="csv")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    paths = (json.loads(Path(args.pdb_manifest).read_text()) if args.pdb_manifest else
             sorted(str(p) for p in Path(args.pdb_dir).rglob("*.pdb")) if args.pdb_dir else [args.pdb_file])
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not Path(p).is_file() for p in paths):
        parser.error("Provide existing PDB files")
    if args.runtime_worker:
        encoder = ESM3_structure_encoder_v0(args.device, args.weights)
        rows = [get_esm3_structure_seq(p, encoder, args.device) for p in paths]
        Path(args.out_file).write_text(json.dumps(rows, allow_nan=False), encoding="utf-8")
        return
    from mcp_tools.models.factory.training.data.esm3_runtime import encode_structures
    rows = encode_structures(paths, args.device)
    if args.out_format == "csv":
        import csv
        with Path(args.out_file).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("name", "esm3_structure_seq", "plddt"))
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, "esm3_structure_seq": json.dumps(row["esm3_structure_seq"])})
    else:
        Path(args.out_file).write_text("\n".join(json.dumps(row, allow_nan=False) for row in rows), encoding="utf-8")


if __name__ == "__main__":
    main()
