import torch
from transformers import (
    EsmTokenizer, EsmModel,
    BertTokenizer, BertModel,
    T5Tokenizer, T5EncoderModel,
    AutoTokenizer, PreTrainedModel,
)
from .adapter_model import AdapterModel
from .lora_model import LoraModel

def create_models(args):
    """Create and initialize models and tokenizer."""
    # Create tokenizer and PLM
    tokenizer, plm_model = create_plm_and_tokenizer(args)
    
    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    
    # Handle structure sequence vocabulary
    if args.training_method == 'ses-adapter':
        args.vocab_size = get_vocab_size(plm_model, args.structure_seq)
    
    # Create adapter model
    model = AdapterModel(args)
    
    # Handle PLM parameters based on training method
    if args.training_method != 'full':
        freeze_plm_parameters(plm_model)
    if args.training_method == 'lora':
        setup_lora_plm(plm_model, args)
    
    # Move models to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    plm_model = plm_model.to(device)
    
    return model, plm_model, tokenizer

def creat_lora_model(args):
    tokenizer, plm_model = create_plm_and_tokenizer(args)
    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)

    # create lora model
    model = LoraModel(args=args)
    # Handle PLM parameters based on training method
    if args.training_method != "full":
        freeze_plm_parameters(plm_model)
    if args.training_method in ["lora", "plm-lora"]:
        setup_lora_plm(plm_model, args)
    print(" Using plm lora ")
    # Move models to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    plm_model = plm_model.to(device)
    return model, plm_model, tokenizer

def freeze_plm_parameters(plm_model):
    """Freeze all parameters in the pre-trained language model."""
    for param in plm_model.parameters():
        param.requires_grad = False
    plm_model.eval()  # Set to evaluation mode

def setup_lora_plm(plm_model, args):
    """Setup LoRA for pre-trained language model."""
    # Import LoRA configurations
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

    if not isinstance(plm_model, PreTrainedModel):
        raise TypeError("based_model must be a PreTrainedModel instance")

    # validate lora_target_modules exist in model
    available_modules = [name for name, _ in plm_model.named_modules()]
    for module in args.lora_target_modules:
        if not any(module in name for name in available_modules):
            raise ValueError(f"Target module {module} not found in model")
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    # Apply LoRA to model
    plm_model = get_peft_model(plm_model, peft_config)
    plm_model.print_trainable_parameters()

def create_plm_and_tokenizer(args):
    """Create pre-trained language model and tokenizer based on model type."""
    if "esm" in args.plm_model:
        tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        plm_model = EsmModel.from_pretrained(args.plm_model)
    elif "bert" in args.plm_model:
        tokenizer = BertTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = BertModel.from_pretrained(args.plm_model)
    elif "prot_t5" in args.plm_model:
        tokenizer = T5Tokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model)
    elif "ankh" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model)
    else:
        raise ValueError(f"Unsupported model type: {args.plm_model}")
    
    return tokenizer, plm_model

def get_hidden_size(plm_model, model_type):
    """Get hidden size based on model type."""
    if "esm" in model_type:
        return plm_model.config.hidden_size
    elif "bert" in model_type:
        return plm_model.config.hidden_size
    elif "prot_t5" in model_type or "ankh" in model_type:
        return plm_model.config.d_model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_vocab_size(plm_model, structure_seq):
    """Get vocabulary size for structure sequences."""
    if 'esm3_structure_seq' in structure_seq:
        return max(plm_model.config.vocab_size, 4100)
    return plm_model.config.vocab_size 