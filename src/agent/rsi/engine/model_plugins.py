"""Trusted model plugin interface shared by both scientific execution modes.

Plugins cannot register their own validation results. The host controller binds
training data, withheld assessments, applicability and acceptance thresholds.
"""

from collections.abc import Callable
from dataclasses import dataclass

from agent.rsi import protein_model


@dataclass(frozen=True)
class ModelPlugin:
    id: str
    fit: Callable
    predict: Callable
    evaluate: Callable
    source_sha256: str
    acceptance_metric: str = "mse_skill_vs_training_mean"
    validate_rows: Callable = protein_model.validate_rows
    evaluate_with_predictions: Callable | None = None


def get_model_plugin(plugin_id: str) -> ModelPlugin:
    from pathlib import Path

    from .storage import sha256_file

    if plugin_id in {
        "generated_context.v1",
        "generated_context.v2",
        "generated_context.v3",
        "generated_assay.v1",
        "generated_antibody.v1",
    }:
        from .context_interfaces import module_for

        context = module_for(plugin_id)

        return ModelPlugin(
            plugin_id,
            context.fit_candidate,
            context.predict,
            context.evaluate,
            context.implementation_hash(),
            context.ACCEPTANCE_METRIC,
            context.validate_rows,
            context.evaluate_with_predictions,
        )
    if plugin_id == "generated_sequence.v1":
        from agent.rsi import generated_sequence as sequence

        return ModelPlugin(
            plugin_id,
            sequence.fit_candidate,
            sequence.predict,
            sequence.evaluate,
            sequence.implementation_hash(),
            sequence.ACCEPTANCE_METRIC,
            sequence.validate_rows,
            sequence.evaluate_with_predictions,
        )
    if plugin_id == "generated_features.v1":
        from agent.rsi import generated_features as features

        return ModelPlugin(
            plugin_id,
            features.fit_candidate,
            features.predict,
            features.evaluate,
            features.implementation_hash(),
            features.ACCEPTANCE_METRIC,
            features.validate_rows,
            features.evaluate_with_predictions,
        )
    if plugin_id == "generated_classifier.large.v1":
        from agent.rsi import generated_classifier_large as classifier

        return ModelPlugin(
            plugin_id,
            classifier.fit_candidate,
            classifier.predict,
            classifier.evaluate,
            classifier.implementation_hash(),
            classifier.ACCEPTANCE_METRIC,
            classifier.validate_rows,
            classifier.evaluate_with_predictions,
        )
    if plugin_id == "generated_classifier.v1":
        from agent.rsi import generated_classifier as classifier

        return ModelPlugin(
            plugin_id,
            classifier.fit_candidate,
            classifier.predict,
            classifier.evaluate,
            classifier.implementation_hash(),
            classifier.ACCEPTANCE_METRIC,
            classifier.validate_rows,
            classifier.evaluate_with_predictions,
        )
    if plugin_id == "generated_python.v1":
        from agent.rsi import generated_model
        from .code_sandbox import implementation_hash

        return ModelPlugin(
            plugin_id,
            generated_model.fit_candidate,
            generated_model.predict,
            generated_model.evaluate,
            implementation_hash(),
            validate_rows=generated_model.validate_rows,
            evaluate_with_predictions=generated_model.evaluate_with_predictions,
        )
    if plugin_id != protein_model.PLUGIN_ID:
        raise ValueError(f"Unsupported trusted model plugin: {plugin_id}")
    return ModelPlugin(
        plugin_id,
        protein_model.fit_candidate,
        protein_model.predict,
        protein_model.evaluate,
        sha256_file(Path(protein_model.__file__)),
        evaluate_with_predictions=protein_model.evaluate_with_predictions,
    )
