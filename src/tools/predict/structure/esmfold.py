"""
ESMFold structure prediction tool: FASTA/sequence -> PDB.
Two backends: local (on-device GPU) and pjlab (DrugSDA remote API).
Usage (from project root):
  python src/tools/esmfold.py --backend local --sequence ... --out_file ...
  python src/tools/esmfold.py --backend pjlab --sequence ...
Default endpoint is read from ESMFOLD_BACKEND in .env (local or pjlab).
"""
import os
import json
import base64
import argparse
import asyncio
import gc
from typing import Optional, Dict, Any, Tuple

import biotite.structure.io as bsio

try:
    from web.utils.common_utils import get_save_path
except ImportError:
    try:
        from src.web.utils.common_utils import get_save_path
    except ImportError:
        get_save_path = None


def _get_default_output_dir() -> str:
    """Default output dir under TEMP_OUTPUTS_DIR."""
    if get_save_path:
        return str(get_save_path("Agent", "ESMFold"))
    return os.path.join(os.getenv("TEMP_OUTPUTS_DIR", "temp_outputs"), "Agent", "ESMFold")


def _get_default_backend() -> str:
    """Load default backend from .env. Returns 'local' or 'pjlab'."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    backend = os.getenv("ESMFOLD_BACKEND", "local").strip().lower()
    return backend if backend in ("local", "pjlab") else "local"


# Local mode imports (lazy load)
def _local_imports():
    import torch
    import pandas as pd
    from tqdm import tqdm
    from Bio import SeqIO
    from transformers import AutoTokenizer, EsmForProteinFolding
    from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    return torch, pd, tqdm, SeqIO, AutoTokenizer, EsmForProteinFolding, to_pdb, OFProtein, atom14_to_atom37


# ---------- Shared utilities ----------
def read_fasta(file_path: str, key: str) -> str:
    from Bio import SeqIO
    return str(getattr(SeqIO.read(file_path, 'fasta'), key))


def read_multi_fasta(file_path: str) -> dict:
    sequences = {}
    current_sequence = ''
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences[header] = current_sequence
                    current_sequence = ''
                header = line
            else:
                current_sequence += line
        if current_sequence:
            sequences[header] = current_sequence
    return sequences


def base64_to_pdb_file(base64_string: str, file_name: str = "protein_structure.pdb",
                       save_dir: str = "./protein_structures") -> Optional[str]:
    """Decode Base64 to PDB and save to file."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        pdb_content = base64.b64decode(base64_string)
        file_path = os.path.join(save_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(pdb_content)
        print(f"✓ PDB saved to: {file_path}")
        return file_path
    except Exception as e:
        print(f"✗ Failed to save PDB: {e}")
        return None


# ---------- Local endpoint ----------
def _convert_outputs_to_pdb(outputs, to_pdb_fn, OFProtein, atom14_to_atom37_fn):
    final_atom_positions = atom14_to_atom37_fn(outputs["positions"][-1], outputs)
    outputs = {k: v.to("cpu").numpy() for k, v in outputs.items()}
    final_atom_positions = final_atom_positions.cpu().numpy()
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in range(outputs["aatype"].shape[0]):
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa,
            atom_positions=pred_pos,
            atom_mask=mask,
            residue_index=resid,
            b_factors=outputs["plddt"][i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb_fn(pred))
    return pdbs


def predict_local(sequence: str, output_dir: str = "./protein_structures",
                  output_file: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Local ESMFold prediction (requires GPU).
    Returns (PDB path, result info) or (None, None).
    """
    try:
        torch, _, _, _, AutoTokenizer, EsmForProteinFolding, to_pdb, OFProtein, atom14_to_atom37 = _local_imports()
    except ImportError as e:
        print(f"✗ Local mode requires PyTorch and transformers: {e}")
        return None, None

    os.makedirs(output_dir, exist_ok=True)
    out_file = output_file or os.path.join(output_dir, "structure.pdb")

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda()
    torch.backends.cuda.matmul.allow_tf32 = True

    try:
        tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()
        with torch.no_grad():
            output = model(tokenized_input)
        pdbs = _convert_outputs_to_pdb(output, to_pdb, OFProtein, atom14_to_atom37)
        with open(out_file, "w") as f:
            f.write("\n".join(pdbs))
        struct = bsio.load_structure(out_file, extra_fields=["b_factor"])
        plddt = float(struct.b_factor.mean())
        result_info = {"local_path": out_file, "sequence_length": len(sequence), "plddt": plddt}
        return out_file, result_info
    except Exception as e:
        print(f"✗ Local prediction failed: {e}")
        return None, None
    finally:
        gc.collect()


# ---------- PJLab / DrugSDA backend ----------
def _get_pjlab_env():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    url_model = os.getenv("DRUGSDA_MODEL_SERVER_URL")
    url_tool = os.getenv("DRUGSDA_TOOL_SERVER_URL")
    return url_model, url_tool


class DrugSDAClient:
    """DrugSDA MCP client for PJLab remote ESMFold prediction."""

    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
        self.session_ctx = None
        self.transport = None

    async def connect(self, verbose: bool = True) -> bool:
        try:
            from mcp.client.streamable_http import streamablehttp_client
            from mcp import ClientSession

            if verbose:
                print(f"Connecting to {self.server_url}...")
            self.transport = streamablehttp_client(
                url=self.server_url,
                headers={"SCP-HUB-API-KEY": os.getenv("SCP_HUB_API_KEY")}
            )
            self.read, self.write, self.get_session_id = await self.transport.__aenter__()
            self.session_ctx = ClientSession(self.read, self.write)
            self.session = await self.session_ctx.__aenter__()
            await self.session.initialize()
            if verbose:
                print(f"✓ Connected (session ID: {self.get_session_id()})")
            return True
        except Exception as e:
            if verbose:
                print(f"✗ Connection failed: {e}")
                import traceback
                traceback.print_exc()
            return False

    async def disconnect(self, verbose: bool = True) -> None:
        try:
            if self.session_ctx:
                await self.session_ctx.__aexit__(None, None, None)
            if hasattr(self, 'transport') and self.transport:
                await self.transport.__aexit__(None, None, None)
            if verbose:
                print("✓ Disconnected")
        except Exception as e:
            if verbose:
                print(f"✗ Disconnect error: {e}")

    def parse_result(self, result: Any) -> Any:
        try:
            if hasattr(result, 'content') and result.content:
                content = result.content[0]
                if hasattr(content, 'text'):
                    return json.loads(content.text)
            return str(result)
        except Exception as e:
            return {"error": f"Parse failed: {e}", "raw": str(result)}


async def predict_protein_structure_pjlab(sequence: str, output_dir: str = "./protein_structures",
                                         verbose: bool = True) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Predict structure via PJLab DrugSDA remote ESMFold API."""
    url_model, url_tool = _get_pjlab_env()
    if not url_model or not url_tool:
        if verbose:
            print("✗ Set DRUGSDA_MODEL_SERVER_URL and DRUGSDA_TOOL_SERVER_URL in .env")
        return None, None

    model_client = DrugSDAClient(url_model)
    if not await model_client.connect(verbose=verbose):
        return None, None

    tool_client = DrugSDAClient(url_tool)
    if not await tool_client.connect(verbose=verbose):
        await model_client.disconnect(verbose=verbose)
        return None, None

    try:
        if verbose:
            print("\nStep 1: ESMFold prediction...")
        result = await model_client.session.call_tool(
            "pred_protein_structure_esmfold",
            arguments={"sequence": sequence}
        )
        result_data = model_client.parse_result(result)
        if "error" in result_data:
            if verbose:
                print(f"✗ Prediction failed: {result_data['error']}")
            return None, None

        pdb_path = result_data.get("pdb_path")
        if verbose:
            print("Server PDB path:", pdb_path)

        if verbose:
            print("\nStep 2: Convert PDB to Base64...")
        result = await tool_client.session.call_tool(
            "server_file_to_base64",
            arguments={"file_path": pdb_path}
        )
        result_data = tool_client.parse_result(result)
        if "error" in result_data:
            if verbose:
                print(f"✗ Conversion failed: {result_data['error']}")
            return None, None

        file_name = result_data.get("file_name", "structure.pdb")
        base64_string = result_data.get("base64_string", "")

        if verbose:
            print("\nStep 3: Save to local...")
        local_file_path = base64_to_pdb_file(base64_string, file_name, output_dir)
        if not local_file_path:
            return None, None

        result_info = {
            "file_name": file_name,
            "server_path": pdb_path,
            "local_path": local_file_path,
            "sequence_length": len(sequence)
        }
        if verbose:
            print(f"\n✓ Success! PDB saved: {local_file_path}")
        return local_file_path, result_info

    except Exception as e:
        if verbose:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
        return None, None
    finally:
        await tool_client.disconnect(verbose=verbose)
        await model_client.disconnect(verbose=verbose)


# ---------- Unified entry ----------
def predict_structure_sync(sequence: str, output_dir: str = "./protein_structures",
                          verbose: bool = True,
                          backend: Optional[str] = None,
                          output_file: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Synchronous structure prediction. Supports local (GPU) and pjlab (DrugSDA remote).
    If backend is None, uses ESMFOLD_BACKEND from .env.
    """
    output_dir = output_dir or _get_default_output_dir()
    if output_dir == "./protein_structures":
        output_dir = _get_default_output_dir()
    backend = backend if backend is not None else _get_default_backend()
    if backend == "local":
        return predict_local(sequence, output_dir=output_dir, output_file=output_file)
    elif backend == "pjlab":
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(predict_protein_structure_pjlab(sequence, output_dir, verbose))
    else:
        print(f"✗ Unknown backend: {backend}, use local or pjlab")
        return None, None


# ---------- Local CLI (batch / single) ----------
def run_local_cli(args: argparse.Namespace) -> None:
    torch, pd, tqdm, SeqIO, AutoTokenizer, EsmForProteinFolding, to_pdb, OFProtein, atom14_to_atom37 = _local_imports()

    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.cuda()
    torch.backends.cuda.matmul.allow_tf32 = True
    if getattr(args, 'fold_chunk_size', None) is not None:
        model.trunk.set_chunk_size(args.fold_chunk_size)

    def convert_outputs(outputs):
        return _convert_outputs_to_pdb(outputs, to_pdb, OFProtein, atom14_to_atom37)

    if args.fasta_file is not None:
        seq_dict = read_multi_fasta(args.fasta_file)
        os.makedirs(args.out_dir, exist_ok=True)
        names, sequences = list(seq_dict.keys()), list(seq_dict.values())
        if getattr(args, 'fasta_chunk_num', None) is not None:
            chunk_size = len(names) // args.fasta_chunk_num + 1
            start = args.fasta_chunk_id * chunk_size
            end = min((args.fasta_chunk_id + 1) * chunk_size, len(names))
            names, sequences = names[start:end], sequences[start:end]
        out_info_dict = {"name": [], "plddt": []}
        bar = tqdm(zip(names, sequences))
        for name, sequence in bar:
            bar.set_description(name)
            name = name[1:].split(" ")[0]
            out_file = os.path.join(args.out_dir, f"{name}.ef.pdb")
            if os.path.exists(out_file):
                out_info_dict["name"].append(name)
                struct = bsio.load_structure(out_file, extra_fields=["b_factor"])
                out_info_dict["plddt"].append(struct.b_factor.mean())
                continue
            try:
                tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()
                with torch.no_grad():
                    output = model(tokenized_input)
            except Exception as e:
                print(e)
                print(f"Failed to predict {name}")
                continue
            gc.collect()
            pdb = convert_outputs(output)
            with open(out_file, "w") as f:
                f.write("\n".join(pdb))
            out_info_dict["name"].append(name)
            struct = bsio.load_structure(out_file, extra_fields=["b_factor"])
            out_info_dict["plddt"].append(struct.b_factor.mean())
        if getattr(args, 'out_info_file', None) is not None:
            pd.DataFrame(out_info_dict).to_csv(args.out_info_file, index=False)

    elif getattr(args, 'fasta_dir', None) is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        proteins = sorted(os.listdir(args.fasta_dir))
        bar = tqdm(proteins)
        for p in bar:
            name = p[:-6]
            bar.set_description(name)
            out_file = os.path.join(args.out_dir, f"{name}.ef.pdb")
            if os.path.exists(out_file):
                continue
            bar.set_description(p)
            sequence = read_fasta(os.path.join(args.fasta_dir, p), "seq")
            tokenized_input = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)['input_ids'].cuda()
            with torch.no_grad():
                output = model(tokenized_input)
            pdb = convert_outputs(output)
            with open(out_file, "w") as f:
                f.write("\n".join(pdb))
            struct = bsio.load_structure(out_file, extra_fields=["b_factor"])
            print(p, struct.b_factor.mean())

    elif args.sequence is not None:
        with torch.no_grad():
            output = model.infer_pdb(args.sequence)
        with open(args.out_file, "w") as f:
            f.write(output)
        struct = bsio.load_structure(args.out_file, extra_fields=["b_factor"])
        print(struct.b_factor.mean())


if __name__ == "__main__":
    default_backend = _get_default_backend()
    parser = argparse.ArgumentParser(description="ESMFold structure prediction: local (GPU) / pjlab (remote)")
    parser.add_argument("--backend", type=str, default=None, choices=["local", "pjlab"],
                        help=f"Backend (default from ESMFOLD_BACKEND={default_backend})")
    parser.add_argument("--sequence", type=str, default=None)
    parser.add_argument("--fasta_file", type=str, default=None)
    parser.add_argument("--fasta_chunk_num", type=int, default=None)
    parser.add_argument("--fasta_chunk_id", type=int, default=None)
    parser.add_argument("--fasta_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--out_file", type=str, default="result.pdb")
    parser.add_argument("--out_info_file", type=str, default=None)
    parser.add_argument("--fold_chunk_size", type=int, default=None)
    args = parser.parse_args()

    backend = args.backend if args.backend is not None else default_backend

    if backend == "pjlab":
        if args.sequence:
            pdb_path, info = predict_structure_sync(args.sequence, output_dir=args.out_dir or "./protein_structures",
                                                    backend="pjlab", verbose=True)
            if pdb_path:
                print(f"Success: {pdb_path}")
                print(info)
            else:
                print("Prediction failed")
        else:
            print("PJLab mode requires --sequence. For batch use: --backend local --fasta_file ...")
    else:
        if not args.out_dir and not args.sequence:
            parser.error("Local mode requires --out_dir (batch) or --sequence --out_file (single)")
        if args.fasta_file or args.fasta_dir or args.sequence:
            if args.fasta_file or args.fasta_dir:
                args.out_dir = args.out_dir or "./esmfold_output"
            run_local_cli(args)
        else:
            parser.error("Specify --sequence, --fasta_file, or --fasta_dir")
