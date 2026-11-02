"""Prefer an already downloaded model over repeated network metadata requests."""
from pathlib import Path


def cached_model(model_id, cache=None):
    if Path(model_id).is_dir():
        return str(model_id)
    from huggingface_hub import try_to_load_from_cache
    config = try_to_load_from_cache(model_id, 'config.json', cache_dir=cache)
    if isinstance(config, str):
        folder = Path(config).parent
        if any((folder / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')):
            return str(folder)
    return str(model_id)
