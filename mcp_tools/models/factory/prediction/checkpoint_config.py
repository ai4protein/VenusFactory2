"""Project checkpoint model settings without changing host-selected I/O.

The full saved configuration remains available for label/scale interpretation
and provenance. Only fields read by AdapterModel or prediction enter the runtime
Namespace; paths, execution options and private attributes stay host-owned.
"""

import copy

MODEL_FIELDS = frozenset({
    "attention_probs_dropout", "hidden_size", "num_attention_head", "num_labels",
    "plm_model", "pooling_dropout", "pooling_method", "problem_type",
    "structure_seq", "training_method", "vocab_size", "dataset",
})


def apply_checkpoint_config(args, config, *, record_snapshot=True):
    """Project model fields; only the actual loader records interpretation evidence."""
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        raise ValueError("Checkpoint configuration must be a JSON object")
    if type(record_snapshot) is not bool:
        raise ValueError("Checkpoint snapshot mode must be explicit")
    snapshot = copy.deepcopy(config) if record_snapshot else None
    for key in MODEL_FIELDS:
        if key in config:
            setattr(args, key, copy.deepcopy(config[key]))
    if record_snapshot:
        args._inference_checkpoint_config = snapshot
    elif hasattr(args, "_inference_checkpoint_config"):
        delattr(args, "_inference_checkpoint_config")
