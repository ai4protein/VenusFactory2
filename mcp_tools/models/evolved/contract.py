"""Public prediction arguments shared by Agent, MCP and queue validation."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


PREDICTION_DESCRIPTION = (
    "Predict with a selected Scientist model. Add inputs to prediction_call.arguments returned by "
    "describe_evolved_model: supply entry_id, scope_ref and inputs together in one argument object. "
    "Use 1-256 unlabelled records matching input_fields/input_contract; preserve IDs and values. "
    "The exact full inline scope can replace scope_ref. Evidence and inputs are revalidated; "
    "this never trains or publishes a model. Local only."
)


class PredictionArguments(BaseModel):
    """Predict a selected model using its described applicability binding."""

    model_config = ConfigDict(
        extra="forbid", strict=True,
        json_schema_extra={
            "anyOf": [
                {"required": ["scope_ref"], "properties": {"scope_ref": {"type": "string"}}},
                {"required": ["scope"], "properties": {"scope": {"type": "object"}}},
            ],
        },
    )
    entry_id: str = Field(pattern=r"^[0-9a-f]{64}$", description="Exact model entry_id returned by discovery.")
    inputs: list[dict] = Field(
        min_length=1, max_length=256,
        description="Unlabelled records matching this model's input_contract; retain IDs and values.",
    )
    scope_ref: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$",
        description="Copy scope_ref from describe_evolved_model. Required unless the complete scope is supplied.",
    )
    scope: dict | None = Field(
        default=None,
        description="Alternative to scope_ref: the exact full scope returned by describe_evolved_model.",
    )

    @model_validator(mode="after")
    def require_scope_binding(self):
        if self.scope_ref is None and self.scope is None:
            raise ValueError("Supply scope_ref or the complete scope.")
        return self
