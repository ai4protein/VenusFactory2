"""
Search: MCP infra only (SCP client + OSS upload).
Backend: local (local paths only, no upload) vs pjlab (upload via SCP, return URLs).
All data/API logic lives in database/* and source/*; this module only wires them + upload.

Functions:
  get_uniprot_sequence(uniprot_id)           -> dict
  call_interpro_function_query(uniprot_id)   -> str (InterPro/GO by UniProt)
  download_pdb_structure_from_id(pdb_id, format, backend) -> str
  download_ncbi_sequence(accession_id, format, backend)   -> str
  download_alphafold_structure(uniprot_id, format, backend) -> str
  upload_file_to_oss_sync(file_path, backend) -> url or None
"""
import json
import os
import sys
sys.path.append(os.getcwd())
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

import requests
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

from src.web.utils.common_utils import get_save_path
from src.tools.search.database.uniprot.uniprot_sequence import download_uniprot_sequence
from src.tools.search.database.interpro.interpro_proteins import query_interpro_by_uniprot
from src.tools.search.database.rcsb.rcsb_structure import download_pdb_by_id
from src.tools.search.database.utils.utils import get_chain_sequences_from_pdb
from src.tools.search.database.ncbi.ncbi_sequence import download_ncbi_fasta
from src.tools.search.database.alphafold.alphafold_structure import download_alphafold_structure

# ---------- Backend and upload (shared with mutation/predict tools_mcp) ----------
BACKEND_LOCAL = "local"
BACKEND_PJLAB = "pjlab"


def _default_backend() -> str:
    v = (os.getenv("AGENT_DEFAULT_BACKEND") or "local").strip().lower()
    return v if v in (BACKEND_LOCAL, BACKEND_PJLAB) else BACKEND_LOCAL


DEFAULT_BACKEND = _default_backend()
SCP_WORKFLOW_SERVER_URL = os.getenv("SCP_WORKFLOW_SERVER_URL", "http://115.190.136.251:8080/mcp")


def get_gradio_base_url(backend: str = None) -> str:
    """Same Gradio URL for local and pjlab; pjlab uses SCP for uploads only."""
    return os.getenv("GRADIO_BASE_URL", "http://localhost:7860/")


class SCPWorkflowClient:
    def __init__(self, server_url: str = SCP_WORKFLOW_SERVER_URL):
        self.server_url = server_url
        self.session = None
        self.transport = None
        self.session_ctx = None

    async def connect(self, timeout: int = 30):
        self.transport = streamablehttp_client(url=self.server_url, sse_read_timeout=60 * 10)
        self.read, self.write, self.get_session_id = await asyncio.wait_for(self.transport.__aenter__(), timeout=timeout)
        self.session_ctx = ClientSession(self.read, self.write)
        self.session = await self.session_ctx.__aenter__()
        await asyncio.wait_for(self.session.initialize(), timeout=timeout)

    async def disconnect(self):
        try:
            if self.session_ctx:
                await self.session_ctx.__aexit__(None, None, None)
            if self.transport:
                await self.transport.__aexit__(None, None, None)
        except Exception:
            pass

    async def generate_presigned_url(self, key: str, expires_seconds: int = 3600) -> Dict[str, Any]:
        result = await self.session.call_tool("generate_presigned_url", arguments={"key": key, "expires_seconds": expires_seconds})
        if hasattr(result, 'content') and result.content:
            return json.loads(result.content[0].text)
        return result


async def upload_file_via_curl(upload_url: str, file_path: str) -> bool:
    try:
        def _put():
            with open(file_path, 'rb') as f:
                return requests.put(upload_url, data=f, timeout=300).status_code
        code = await asyncio.to_thread(_put)
        return code in [200, 201]
    except Exception as e:
        print(f"Upload failed: {e}")
        return False


async def upload_file_to_cloud_async(file_path: str, key: Optional[str] = None, expires_seconds: int = 3600, backend: str = BACKEND_PJLAB) -> Optional[str]:
    if backend == BACKEND_LOCAL:
        return None
    client = SCPWorkflowClient()
    try:
        await client.connect()
        key = key or Path(file_path).name
        result = await client.generate_presigned_url(key, expires_seconds)
        if "error" in result:
            return None
        success = await upload_file_via_curl(result["upload"]["url"], file_path)
        return result["download"]["url"] if success else None
    except Exception as e:
        print(f"Upload error: {e}")
        return None
    finally:
        await client.disconnect()


def upload_file_to_oss_sync(file_path: str, backend: str = None) -> Optional[str]:
    if backend is None:
        backend = DEFAULT_BACKEND
    if backend == BACKEND_LOCAL or not file_path or not os.path.exists(file_path):
        return None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(upload_file_to_cloud_async(file_path, backend=backend))
        finally:
            loop.close()
    except Exception as e:
        print(f"Upload failed {file_path}: {e}")
        return None





def get_uniprot_sequence(uniprot_id: str) -> dict:
    """Fetch protein sequence for one UniProt ID (from database/uniprot)."""
    return download_uniprot_sequence(uniprot_id)


def call_interpro_function_query(uniprot_id: str) -> str:
    """InterPro/GO query by UniProt ID (from database/interpro)."""
    try:
        return query_interpro_by_uniprot(uniprot_id)
    except Exception as e:
        return json.dumps({"success": False, "uniprot_id": uniprot_id, "error_message": str(e)}, indent=4)


def download_pdb_structure_from_id(pdb_id: str, output_format: str = "pdb", backend: str = None) -> str:
    backend = backend or DEFAULT_BACKEND
    try:
        structure_dir = get_save_path("Download_Data", "RCSB")
        structure_dir.mkdir(parents=True, exist_ok=True)
        file_path = download_pdb_by_id(pdb_id, str(structure_dir), file_type=output_format, unzip=True)
        if not file_path or not os.path.exists(file_path):
            return json.dumps({"success": False, "pdb_id": pdb_id, "error_message": "Download failed or file not found"})
        seqs = get_chain_sequences_from_pdb(file_path)
        if not seqs:
            return json.dumps({"success": False, "pdb_id": pdb_id, "error_message": "No protein sequence found in PDB file."})
        pdb_file_url = upload_file_to_oss_sync(file_path, backend=backend)
        return json.dumps({
            "success": True,
            "pdb_id": pdb_id.upper(),
            "pdb_file": file_path,
            "oss_url": pdb_file_url,
            "sequences": seqs,
            "message": f"PDB structure downloaded: {file_path}",
        }, indent=2)
    except Exception as e:
        return json.dumps({"success": False, "pdb_id": pdb_id, "error_message": str(e)})


def download_ncbi_sequence(accession_id: str, output_format: str = "fasta", backend: str = None) -> str:
    backend = backend or DEFAULT_BACKEND
    try:
        sequence_dir = get_save_path("Download_Data", "NCBI")
        sequence_dir.mkdir(parents=True, exist_ok=True)
        db_type = "protein" if accession_id.startswith(("NP_", "XP_", "YP_", "WP_")) else "nuccore"
        _, message, _ = download_ncbi_fasta(accession_id, db_type, str(sequence_dir), merge_output=False)
        expected_file = Path(sequence_dir) / f"{accession_id}.fasta"
        if expected_file.exists():
            url = upload_file_to_oss_sync(str(expected_file), backend=backend)
            return json.dumps({
                "success": True,
                "accession_id": accession_id,
                "format": output_format,
                "fasta_file": str(expected_file),
                "oss_url": url,
                "message": f"Sequence downloaded: {expected_file}",
            })
        return json.dumps({"success": False, "accession_id": accession_id, "error_message": message})
    except Exception as e:
        return json.dumps({"success": False, "accession_id": accession_id, "error_message": str(e)})


def download_alphafold_structure(uniprot_id: str, output_format: str = "pdb", backend: str = None) -> str:
    backend = backend or DEFAULT_BACKEND
    try:
        structure_dir = get_save_path("Download_Data", "AlphaFold")
        structure_dir.mkdir(parents=True, exist_ok=True)
        ok, file_path = download_alphafold_structure(uniprot_id, str(structure_dir))
        if not ok or not file_path or not os.path.exists(file_path):
            return json.dumps({"success": False, "uniprot_id": uniprot_id, "error_message": "Download failed or file not found"})
        confidence_info = {}
        try:
            content = Path(file_path).read_text()
            scores = [float(line[60:66].strip()) for line in content.split("\n") if line.startswith("ATOM") and "CA" in line]
            if scores:
                confidence_info = {
                    "mean_confidence": round(sum(scores) / len(scores), 2),
                    "min_confidence": round(min(scores), 2),
                    "max_confidence": round(max(scores), 2),
                    "high_confidence_residues": sum(1 for s in scores if s >= 70),
                    "total_residues": len(scores),
                }
        except Exception:
            pass
        url = upload_file_to_oss_sync(file_path, backend=backend)
        return json.dumps({
            "success": True,
            "uniprot_id": uniprot_id,
            "format": output_format,
            "pdb_file": file_path,
            "oss_url": url,
            "confidence_info": confidence_info,
            "message": f"AlphaFold structure downloaded: {file_path}",
        })
    except Exception as e:
        return json.dumps({"success": False, "uniprot_id": uniprot_id, "error_message": str(e)})
