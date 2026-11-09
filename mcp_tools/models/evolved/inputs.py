"""Early input feedback from the model's existing scientific validator."""
from __future__ import annotations

from mcp_tools.common.tool_arguments import ToolArgumentError

# Only known, value-free diagnostics cross the public boundary. Scientific
# rules remain in the frozen model interfaces, including sequence/assay models.
_FEEDBACK = {
    "Observation protein differs from model scope":
        "Input protein differs from the selected model. Search the RSI category using the input protein as query, then inspect the matching model.",
    "Covariates must match the declared column order":
        "Match the number and order of values to this model's declared covariate columns.",
    "Finite covariates inside their declared domains required":
        "Use finite covariate values inside the model's declared ranges and units.",
    "Binary condition covariates require zero or one":
        "Binary condition values must be zero or one.",
    "Unique observation IDs and independent group identifiers required":
        "Use unique observation IDs and the recorded sample-group identities.",
}


class ModelInputError(ToolArgumentError):
    """Safe scientific-input feedback, distinct from model-evidence failure."""


def validate_model_inputs(interface, scope, inputs):
    from agent.rsi.datasets.model_tasks import inference_inputs
    from agent.rsi.engine.model_plugins import get_model_plugin

    plugin = get_model_plugin(interface)
    try:
        # Use the same label-free projection as actual library inference.
        rows = inference_inputs(inputs, interface)
        plugin.validate_rows(rows, scope, labels=False)
    except (ValueError, KeyError, TypeError) as exc:
        message = _FEEDBACK.get(str(exc)) if isinstance(exc, ValueError) else None
        raise ModelInputError(message or (
            "Inputs do not match the selected model's input contract. Check required fields, "
            "sequence or feature format, identities and applicable conditions."
        )) from None
