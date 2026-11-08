from langchain_core.tools import tool
from pydantic import BaseModel, Field
from mcp_tools.models.pretrained.admet.service import predict_admet


class ADMETInput(BaseModel):
    smiles: list[str] = Field(min_length=1, max_length=1000, description='Small-molecule SMILES strings; online maximum 32 molecules, local maximum 1000.')
    include_physchem: bool = True


@tool('predict_admet', args_schema=ADMETInput)
def predict_admet_tool(smiles: list[str], include_physchem: bool=True) -> dict:
    """Predict small-molecule ADMET: absorption, BBB distribution, CYP metabolism, clearance, hERG, AMES, liver injury and toxicity, plus molecular descriptors. 输入小分子 SMILES，预测吸收、分布、代谢、排泄、毒性和理化性质。Uses local ADMET-AI CPU weights, no external API. Probabilities are not safety scores; returns assay units, preview and CSV/JSON artifacts. Not for protein sequence ADMET."""
    return predict_admet(smiles,include_physchem)
