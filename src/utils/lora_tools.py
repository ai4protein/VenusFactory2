import torch
from transformers import AutoTokenizer, EsmModel, T5Tokenizer, T5EncoderModel, BertModel
from transformers import BertTokenizer, EsmTokenizer, T5Tokenizer
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from typing import List, Dict, Any, Tuple
from transformers import PreTrainedModel


def prepare_for_lora_model(
    based_model,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: List[str,] = ["key", "query", "value"],
):
    if not isinstance(based_model, PreTrainedModel):
        raise TypeError("based_model must be a PreTrainedModel instance")

    # validate target_modules exist in model
    available_modules = [name for name, _ in based_model.named_modules()]
    for module in target_modules:
        if not any(module in name for name in available_modules):
            raise ValueError(f"Target module {module} not found in model")
    # get lora config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    # get lora model
    model = get_peft_model(based_model, lora_config)
    print("Lora model is ready! num of trainable_parameters: ")
    model.print_trainable_parameters()
    return model


def load_lora_model(base_model, lora_ckpt_path):
    model = PeftModel.from_pretrained(base_model, lora_ckpt_path)
    return model


def load_eval_base_model(plm_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "esm" in plm_model:
        base_model = EsmModel.from_pretrained(plm_model).to(device)
    elif "bert" in plm_model:
        base_model = BertModel.from_pretrained(plm_model).to(device)
    elif "prot_t5" in plm_model:
        base_model = T5EncoderModel.from_pretrained(plm_model).to(device)
    elif "ankh" in plm_model:
        base_model = T5EncoderModel.from_pretrained(plm_model).to(device)

    return base_model


def check_lora_params(model):
    lora_params = [
        (name, param) for name, param in model.named_parameters() if "lora_" in name
    ]
    print(f"\n num of lora params: {len(lora_params)}")

    if len(lora_params) == 0:
        print("warning: no lora params found!")
    else:
        print("\n first lora param:")
        name, param = lora_params[0]
        print(f"name: {name}")
        print(f"param.shape: {param.shape}")
        print(f"param.dtype: {param.dtype}")
        print(f"param.device: {param.device}")
        # print(f"param_value:\n{param.data.cpu().numpy()}")
        print(f"requires_grad: {param.requires_grad}")
