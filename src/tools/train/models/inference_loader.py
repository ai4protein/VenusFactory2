"""
Unified inference model loader for predict (single + batch) and evaluate.
Uses model_factory.create_plm_and_tokenizer and loads adapter/LoRA checkpoints in one place.
"""
import os
import json
import torch
from .model_factory import (
    create_plm_and_tokenizer,
    get_hidden_size,
    get_vocab_size,
)
from .adapter_model import AdapterModel
from .lora_model import LoraModel
from peft import PeftModel


def _resolve_model_path(args):
    if getattr(args, "model_path", None):
        return args.model_path
    return os.path.join(
        getattr(args, "output_root", "result"),
        getattr(args, "output_dir", ""),
        getattr(args, "output_model_name", "model.pt"),
    )


def _load_config_into_args(args, config_path):
    """Load config.json and update args for inference."""
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        for key in (
            "pooling_method",
            "problem_type",
            "num_labels",
            "num_attention_head",
            "attention_probs_dropout",
            "pooling_dropout",
        ):
            if key in config:
                setattr(args, key, config[key])
        return True
    except FileNotFoundError:
        return False


def _normalize_structure_seq_args(args):
    """Set structure_seq default and derive use_foldseek / use_ss8."""
    if getattr(args, "structure_seq", None) is None:
        args.structure_seq = ""
    if "foldseek_seq" in args.structure_seq:
        args.use_foldseek = True
    if "ss8_seq" in args.structure_seq:
        args.use_ss8 = True
    structure_seq_list = []
    if getattr(args, "use_foldseek", False) and "foldseek_seq" not in args.structure_seq:
        structure_seq_list.append("foldseek_seq")
    if getattr(args, "use_ss8", False) and "ss8_seq" not in args.structure_seq:
        structure_seq_list.append("ss8_seq")
    if structure_seq_list and not args.structure_seq:
        args.structure_seq = ",".join(structure_seq_list)


def load_inference_model(args, verbose=True):
    """
    Load tokenizer, PLM, adapter/LoRA model and checkpoint for inference.
    Returns (model, plm_model, tokenizer, device).
    For ProPrime_650M_OGT (no adapter), model is None.
    """
    if verbose:
        print("---------- Loading Model and Tokenizer ----------")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if getattr(torch.backends.mps, "is_available", lambda: False)()
        and getattr(torch.backends.mps, "is_built", lambda: False)()
        else "cpu"
    )

    model_path = _resolve_model_path(args)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    config_name = model_path.split("/")[-1].replace(".pt", ".json")
    config_path = os.path.join(os.path.dirname(model_path), config_name)
    if _load_config_into_args(args, config_path) and verbose:
        print(f"Loaded configuration from {config_path}")
    elif verbose:
        print(f"Model config not found at {config_path}. Using command line arguments.")

    _normalize_structure_seq_args(args)

    # ProPrime_650M_OGT: no adapter, return (None, plm_model, tokenizer, device)
    if "ProPrime_650M_OGT" in getattr(args, "plm_model", ""):
        tokenizer, plm_model = create_plm_and_tokenizer(args)
        args.hidden_size = plm_model.config.hidden_size
        args.vocab_size = getattr(plm_model.config, "vocab_size", None)
        plm_model.to(device).eval()
        if verbose:
            print("ProPrime_650M_OGT: adapter not used.")
        return None, plm_model, tokenizer, device

    tokenizer, plm_model = create_plm_and_tokenizer(args)
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    args.vocab_size = getattr(plm_model.config, "vocab_size", None)
    if getattr(args, "eval_method", None) == "ses-adapter":
        args.vocab_size = get_vocab_size(plm_model, getattr(args, "structure_seq", "") or "")

    if verbose:
        eval_method = getattr(args, "eval_method", "freeze")
        print(f"Training method: {eval_method}")
        print(f"Structure sequence: {getattr(args, 'structure_seq', '')}")
        print(f"Use foldseek: {getattr(args, 'use_foldseek', False)}")
        print(f"Use ss8: {getattr(args, 'use_ss8', False)}")
        print(f"Problem type: {getattr(args, 'problem_type', '')}")
        print(f"Number of labels: {getattr(args, 'num_labels', 2)}")
        print(f"Number of attention heads: {getattr(args, 'num_attention_head', 8)}")

    eval_method = getattr(args, "eval_method", "freeze")
    if eval_method in ["full", "ses-adapter", "freeze"]:
        model = AdapterModel(args)
    elif eval_method in ["plm-lora", "plm-qlora", "plm-dora", "plm-adalora", "plm-ia3"]:
        model = LoraModel(args)
    else:
        raise ValueError(f"Unsupported eval_method: {eval_method}")

    if eval_method == "full":
        model_weights = torch.load(model_path, map_location=device)
        model.load_state_dict(model_weights["model_state_dict"])
        plm_model.load_state_dict(model_weights["plm_state_dict"])
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device).eval()

    if eval_method == "plm-lora":
        plm_model = PeftModel.from_pretrained(plm_model, model_path.replace(".pt", "_lora"))
        plm_model = plm_model.merge_and_unload()
    elif eval_method == "plm-qlora":
        plm_model = PeftModel.from_pretrained(plm_model, model_path.replace(".pt", "_qlora"))
        plm_model = plm_model.merge_and_unload()
    elif eval_method == "plm-dora":
        plm_model = PeftModel.from_pretrained(plm_model, model_path.replace(".pt", "_dora"))
        plm_model = plm_model.merge_and_unload()
    elif eval_method == "plm-adalora":
        plm_model = PeftModel.from_pretrained(plm_model, model_path.replace(".pt", "_adalora"))
        plm_model = plm_model.merge_and_unload()
    elif eval_method == "plm-ia3":
        plm_model = PeftModel.from_pretrained(plm_model, model_path.replace(".pt", "_ia3"))
        plm_model = plm_model.merge_and_unload()

    plm_model.to(device).eval()
    return model, plm_model, tokenizer, device
