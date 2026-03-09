"""
Predict MCP: FastMCP tools calling features_operations, finetuned_operations, strcutrue_operations.
"""
from typing import Optional

from fastmcp import FastMCP

# ---------- FastMCP: operations (features, finetuned, structure) ----------
from .features.features_operations import (
    calculate_physchem_from_fasta,
    calculate_rsa_from_pdb,
    calculate_sasa_from_pdb,
    calculate_ss_from_pdb,
)
from .finetuned.fintuned_operations import (
    predict_protein_function,
    predict_residue_function,
)
from .structure.strcutrue_operations import predict_structure_esmfold


mcp = FastMCP("Venus_Predict_MCP")


@mcp.tool(name="calculate_physchem_from_fasta")
def mcp_calculate_physchem_from_fasta(
    fasta_file: str,
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate physicochemical properties from FASTA. Returns status JSON with file_info."""
    return calculate_physchem_from_fasta(fasta_file, output_file=output_file, out_dir=out_dir)


@mcp.tool(name="calculate_rsa_from_pdb")
def mcp_calculate_rsa_from_pdb(
    pdb_file: str,
    chain_id: str = "A",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate RSA per residue from PDB. Returns status JSON with file_info."""
    return calculate_rsa_from_pdb(pdb_file, chain_id=chain_id, output_file=output_file, out_dir=out_dir)


@mcp.tool(name="calculate_sasa_from_pdb")
def mcp_calculate_sasa_from_pdb(
    pdb_file: str,
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate SASA per residue from PDB. Returns status JSON with file_info."""
    return calculate_sasa_from_pdb(pdb_file, output_file=output_file, out_dir=out_dir)


@mcp.tool(name="calculate_ss_from_pdb")
def mcp_calculate_ss_from_pdb(
    pdb_file: str,
    chain_id: str = "A",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Calculate secondary structure per residue from PDB. Returns status JSON with file_info."""
    return calculate_ss_from_pdb(pdb_file, chain_id=chain_id, output_file=output_file, out_dir=out_dir)



@mcp.tool(name="predict_protein_function")
def mcp_predict_protein_function(
    fasta_file: str,
    task: str,
    model_name: str = "Ankh-large",
    adapter_path: Optional[str] = None,
    ckpt_base: str = "ckpt",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Run finetuned protein function prediction (e.g. Solubility, Optimal Temperature). Returns status JSON with file_info."""
    return predict_protein_function(
        fasta_file=fasta_file,
        task=task,
        model_name=model_name,
        adapter_path=adapter_path,
        ckpt_base=ckpt_base,
        output_file=output_file,
        out_dir=out_dir,
    )


@mcp.tool(name="predict_residue_function")
def mcp_predict_residue_function(
    fasta_file: str,
    task: str,
    model_name: str = "ESM2-650M",
    adapter_path: Optional[str] = None,
    ckpt_base: str = "ckpt",
    output_file: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> str:
    """Run finetuned residue prediction (Activity Site, Binding Site, Conserved Site, Motif). Returns status JSON with file_info."""
    return predict_residue_function(
        fasta_file=fasta_file,
        task=task,
        model_name=model_name,
        adapter_path=adapter_path,
        ckpt_base=ckpt_base,
        output_file=output_file,
        out_dir=out_dir,
    )


@mcp.tool(name="predict_structure_esmfold")
def mcp_predict_structure_esmfold(
    sequence: str,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Predict protein structure with ESMFold (local). Returns status JSON with file_info (PDB path)."""
    return predict_structure_esmfold(
        sequence=sequence,
        output_dir=output_dir,
        output_file=output_file,
        verbose=verbose,
    )
