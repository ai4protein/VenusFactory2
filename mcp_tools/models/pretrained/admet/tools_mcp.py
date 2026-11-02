from fastmcp import FastMCP
from mcp_tools.models.pretrained.admet.service import predict_admet

mcp=FastMCP('Venus_ADMET')


@mcp.tool(name='predict_admet')
def admet_prediction(smiles: list[str], include_physchem: bool=True) -> dict:
    """Predict small-molecule ADMET (吸收、分布、代谢、排泄、毒性), hERG, AMES, DILI, BBB and CYP properties from SMILES using local ADMET-AI CPU models. Online maximum 32 molecules, local 1000. Returns endpoint units/species and CSV/JSON artifacts. Classification probabilities are not a safety score. Not a protein sequence predictor."""
    return predict_admet(smiles,include_physchem)
