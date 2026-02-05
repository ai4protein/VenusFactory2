import os
import re
import torch
import yaml
from transformers import (
    EsmTokenizer, EsmModel,
    BertTokenizer, BertModel,
    T5Tokenizer, T5EncoderModel,
    AutoTokenizer, PreTrainedModel,
    AutoModelForMaskedLM, AutoModel
)
from peft import prepare_model_for_kbit_training
from .adapter_model import AdapterModel
from .lora_model import LoraModel


class ProtSSNEmbeddingWrapper(torch.nn.Module):
    """Wraps ProtSSN to produce batched protein-level and residue-level embeddings from PDB paths."""

    def __init__(self, protssn, device=None):
        super().__init__()
        self.protssn = protssn
        self.device = device or next(protssn.gnn_model.parameters()).device

    def eval(self):
        for m in [self.protssn.plm_model, self.protssn.gnn_model]:
            m.eval()
        return self

    def train(self, mode=True):
        # ProtSSN is always frozen; keep eval
        for m in [self.protssn.plm_model, self.protssn.gnn_model]:
            m.eval()
        return self

    @torch.no_grad()
    def get_embeddings(self, pdb_paths):
        """
        Returns:
            protein_embeds: (batch_size, hidden_size)
            residue_embeds: (batch_size, max_len, hidden_size)
            attention_mask: (batch_size, max_len), 1 for valid residue, 0 for pad
        """
        residue_list = []
        lengths = []
        for path in pdb_paths:
            emb = self.protssn.compute_embedding(path, reduction=None)
            if emb is None:
                raise RuntimeError(f"ProtSSN returned None for {path}")
            residue_list.append(emb)
            lengths.append(emb.shape[0])
        max_len = max(lengths)
        hidden_size = residue_list[0].shape[-1]
        batch_size = len(pdb_paths)
        device = residue_list[0].device
        residue_padded = torch.zeros(batch_size, max_len, hidden_size, device=device, dtype=residue_list[0].dtype)
        attention_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        protein_embeds = []
        for i, (emb, L) in enumerate(zip(residue_list, lengths)):
            residue_padded[i, :L] = emb
            attention_mask[i, :L] = 1
            protein_embeds.append(emb.mean(dim=0))
        protein_embeds = torch.stack(protein_embeds, dim=0)
        return protein_embeds, residue_padded, attention_mask


def create_models(args):
    """Create and initialize models and tokenizer."""
    if "ProtSSN" in args.plm_model and args.training_method != "freeze":
        raise ValueError("ProtSSN only supports training_method='freeze' (no LoRA or full fine-tuning).")
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
    if "ProtSSN" in args.plm_model:
        pass  # ProtSSN is always frozen; no PEFT
    elif args.training_method == 'plm-lora':
        plm_model=setup_lora_plm(plm_model, args)
    elif args.training_method == 'plm-qlora':
        plm_model=create_qlora_model(plm_model, args)
    elif args.training_method == 'plm-adalora':
        plm_model=create_adalora_model(plm_model, args)
    elif args.training_method == "plm-dora":
        plm_model=create_dora_model(plm_model, args)
    elif args.training_method == "plm-ia3":
        plm_model=create_ia3_model(plm_model, args)
    
    # Move models to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    plm_model = plm_model.to(device)
    
    return model, plm_model, tokenizer

def create_lora_model(args):
    tokenizer, plm_model = create_plm_and_tokenizer(args)
    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    model = LoraModel(args=args)
    # Enable gradient checkpointing
    plm_model.gradient_checkpointing_enable()
    plm_model=setup_lora_plm(plm_model, args)
    return model, plm_model, tokenizer

def create_qlora_model(args):
    qlora_config = setup_quantization_config()
    tokenizer, plm_model = create_plm_and_tokenizer(args, qlora_config=qlora_config)
    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    model = LoraModel(args=args)
    # Enable gradient checkpointing
    plm_model.gradient_checkpointing_enable()
    plm_model = prepare_model_for_kbit_training(plm_model)
    plm_model=setup_lora_plm(plm_model, args)
    return model, plm_model, tokenizer

def create_dora_model(args):
    tokenizer, plm_model = create_plm_and_tokenizer(args)
    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    model = LoraModel(args=args)
    # Enable gradient checkpointing
    plm_model.gradient_checkpointing_enable()
    plm_model=setup_dora_plm(plm_model, args)
    return model, plm_model, tokenizer

def create_adalora_model(args):
    tokenizer, plm_model = create_plm_and_tokenizer(args)
    # Update hidden size based on PLM
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    model = LoraModel(args=args)
    # Enable gradient checkpointing
    plm_model.gradient_checkpointing_enable()
    plm_model=setup_adalora_plm(plm_model, args)
    print(" Using plm adalora ")
    return model, plm_model, tokenizer

def create_ia3_model(args):
    tokenizer, plm_model = create_plm_and_tokenizer(args)
    args.hidden_size = get_hidden_size(plm_model, args.plm_model)
    model = LoraModel(args=args)
    plm_model.gradient_checkpointing_enable()
    plm_model = prepare_model_for_kbit_training(plm_model)
    plm_model=setup_ia3_plm(plm_model, args)
    print(" Using plm IA3 ")
    return model, plm_model, tokenizer

def lora_factory(args):
    if args.training_method in "plm-lora":
        model, plm_model, tokenizer = create_lora_model(args)
    elif args.training_method == "plm-qlora":
        model, plm_model, tokenizer = create_qlora_model(args)
    elif args.training_method == "plm-dora":
        model, plm_model, tokenizer = create_dora_model(args)
    elif args.training_method == "plm-adalora":
        model, plm_model, tokenizer = create_adalora_model(args)
    elif args.training_method == "plm-ia3":
        model, plm_model, tokenizer = create_ia3_model(args)
    else:
        raise ValueError(f"Unsupported lora training method: {args.training_method}")
    # Move models to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    plm_model = plm_model.to(device)
    return model, plm_model, tokenizer

def freeze_plm_parameters(plm_model):
    """Freeze all parameters in the pre-trained language model."""
    if isinstance(plm_model, ProtSSNEmbeddingWrapper):
        for m in [plm_model.protssn.plm_model, plm_model.protssn.gnn_model]:
            for param in m.parameters():
                param.requires_grad = False
            m.eval()
        return
    for param in plm_model.parameters():
        param.requires_grad = False
    plm_model.eval()  # Set to evaluation mode

def setup_quantization_config():
    """Setup quantization configuration."""
    from transformers import BitsAndBytesConfig
    # https://huggingface.co/docs/peft/v0.14.0/en/developer_guides/quantization#quantize-a-model
    qlora_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    return qlora_config

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
    return plm_model

def setup_dora_plm(plm_model, args):
    """Setup DoRA for pre-trained language model."""
    # Import DoRA configurations
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

    if not isinstance(plm_model, PreTrainedModel):
        raise TypeError("based_model must be a PreTrainedModel instance")

    # validate Dora_target_modules exist in model
    available_modules = [name for name, _ in plm_model.named_modules()]
    for module in args.lora_target_modules:
        if not any(module in name for name in available_modules):
            raise ValueError(f"Target module {module} not found in model")
    # Configure DoRA
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        use_dora=True
    )
    # Apply DoRA to model
    plm_model = get_peft_model(plm_model, peft_config)
    plm_model.print_trainable_parameters()
    return plm_model

def setup_adalora_plm(plm_model, args):
    """Setup AdaLoRA for pre-trained language model."""
    # Import AdaLoRA configurations
    from peft import get_peft_config, get_peft_model, AdaLoraConfig, TaskType

    if not isinstance(plm_model, PreTrainedModel):
        raise TypeError("based_model must be a PreTrainedModel instance")

    # validate lora_target_modules exist in model
    available_modules = [name for name, _ in plm_model.named_modules()]
    for module in args.lora_target_modules:
        if not any(module in name for name in available_modules):
            raise ValueError(f"Target module {module} not found in model")
        
    # Configure AdaLoRA
    peft_config = AdaLoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        peft_type="ADALORA",
        init_r=12,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules
    )
    # Apply AdaLoRA to model
    plm_model = get_peft_model(plm_model, peft_config)
    plm_model.print_trainable_parameters()
    return plm_model

def setup_ia3_plm(plm_model, args):
    """Setup IA3 for pre-trained language model."""
    # Import LoRA configurations
    from peft import IA3Model, IA3Config, get_peft_model, TaskType

    if not isinstance(plm_model, PreTrainedModel):
        raise TypeError("based_model must be a PreTrainedModel instance")

    # validate lora_target_modules exist in model
    available_modules = [name for name, _ in plm_model.named_modules()]
    print(available_modules)
    for module in args.lora_target_modules:
        if not any(module in name for name in available_modules):
            raise ValueError(f"Target module {module} not found in model")
    # Configure LoRA
    peft_config = IA3Config(
        task_type=TaskType.FEATURE_EXTRACTION,
        peft_type="IA3",
        target_modules=args.lora_target_modules,
        feedforward_modules=args.feedforward_modules
    )
    # Apply LoRA to model
    plm_model = get_peft_model(plm_model, peft_config)
    plm_model.print_trainable_parameters()
    return plm_model

def create_plm_and_tokenizer(args, qlora_config=None):
    """Create pre-trained language model and tokenizer based on model type."""
    if "esm" in args.plm_model:
        tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        if qlora_config: 
            plm_model = EsmModel.from_pretrained(args.plm_model, quantization_config=qlora_config) 
        else:
            plm_model = EsmModel.from_pretrained(args.plm_model)
    elif "bert" in args.plm_model:
        tokenizer = BertTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        if qlora_config:
            plm_model = BertModel.from_pretrained(args.plm_model, quantization_config=qlora_config)
        else:
            plm_model = BertModel.from_pretrained(args.plm_model)
    elif "prot_t5" in args.plm_model:
        tokenizer = T5Tokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        if qlora_config:
            plm_model = T5EncoderModel.from_pretrained(args.plm_model, quantization_config=qlora_config)
        else:
            plm_model = T5EncoderModel.from_pretrained(args.plm_model)
    elif "ankh" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        if qlora_config:
            plm_model = T5EncoderModel.from_pretrained(args.plm_model, quantization_config=qlora_config)
        else:
            plm_model = T5EncoderModel.from_pretrained(args.plm_model)
    elif "ProSST" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        if qlora_config:
            plm_model = AutoModelForMaskedLM.from_pretrained(args.plm_model, trust_remote_code=True, quantization_config=qlora_config)
        else:
            plm_model = AutoModelForMaskedLM.from_pretrained(args.plm_model, trust_remote_code=True)
    elif "SaProt" in args.plm_model:
        # SaProt: ESM-style tokenizer + EsmForMaskedLM; input = uppercase aa + lowercase foldseek per residue
        tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        if qlora_config:
            plm_model = AutoModelForMaskedLM.from_pretrained(args.plm_model, quantization_config=qlora_config)
        else:
            plm_model = AutoModelForMaskedLM.from_pretrained(args.plm_model)
    elif "Prime" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, trust_remote_code=True, do_lower_case=False)
        if qlora_config:
            plm_model = AutoModel.from_pretrained(args.plm_model, trust_remote_code=True, quantization_config=qlora_config)
        else:
            plm_model = AutoModel.from_pretrained(args.plm_model, trust_remote_code=True)
    elif "deep" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        if qlora_config:
            plm_model = AutoModel.from_pretrained(args.plm_model, trust_remote_code=True, quantization_config=qlora_config)
        else:
            plm_model = AutoModel.from_pretrained(args.plm_model, trust_remote_code=True)
    elif "ProtSSN" in args.plm_model:
        if qlora_config:
            raise ValueError("ProtSSN does not support quantization (freeze only).")
        tokenizer, plm_model = _create_protssn_wrapper(args)
    else:
        raise ValueError(f"Unsupported model type: {args.plm_model}")
    
    return tokenizer, plm_model


def _parse_protssn_config(plm_model: str):
    """Parse plm_model e.g. ProtSSN-k10_h512 -> (k=10, h=512). Default (10, 512) if no match."""
    m = re.search(r"ProtSSN-k(\d+)_h(\d+)", plm_model, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 10, 512


def _create_protssn_wrapper(args):
    """Build ProtSSN (ESM + GNN) and wrap for embedding extraction. Freeze only.
    Model name (e.g. ProtSSN-k10_h512) determines c_alpha_max_neighbors=k and hidden_channels=h."""
    from src.mutation.models.protssn import ProtSSN, PLM_model, GNN_model
    from src.mutation.utils import NormalizeProtein

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k, h = _parse_protssn_config(args.plm_model)
    # If name has no -k*_h* (e.g. plain "ProtSSN"), use CLI c_alpha_max_neighbors and default h=512
    if "-k" not in args.plm_model.lower() or "_h" not in args.plm_model:
        k = getattr(args, "c_alpha_max_neighbors", 10)

    gnn_config_path = getattr(args, "gnn_config", None) or "src/mutation/models/egnn/egnn.yaml"
    gnn_config = yaml.load(open(gnn_config_path), Loader=yaml.FullLoader)["egnn"]
    gnn_config["hidden_channels"] = h

    gnn_model_path = getattr(args, "gnn_model_path", None)
    if gnn_model_path is None:
        base = os.path.expanduser("~/.cache/huggingface/hub/models--tyang816--ProtSSN/model")
        ckpt_name = f"protssn_k{k}_h{h}.pt"
        if not os.path.exists(os.path.join(base, ckpt_name)):
            os.makedirs(base, exist_ok=True)
            os.system(f"wget -q https://huggingface.co/tyang816/ProtSSN/resolve/main/ProtSSN.zip -P {base} 2>/dev/null || true")
            if os.path.exists(os.path.join(base, "ProtSSN.zip")):
                os.system(f"unzip -o {base}/ProtSSN.zip -d {base} && rm {base}/ProtSSN.zip")
        gnn_model_path = base

    class ProtSSNArgs:
        def __init__(self):
            self.gnn_config = gnn_config
            self.noise_type = None
            self.noise_ratio = 0.0
            self.c_alpha_max_neighbors = k

    plm_name = "facebook/esm2_t33_650M_UR50D"
    esm_model = EsmModel.from_pretrained(plm_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(plm_name)
    protssn_args = ProtSSNArgs()
    plm_module = PLM_model(protssn_args, esm_model, tokenizer)
    gnn_module = GNN_model(protssn_args)
    ckpt = os.path.join(gnn_model_path, f"protssn_k{k}_h{h}.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(
            f"ProtSSN GNN checkpoint not found: {ckpt}. "
            f"Use plm_model like ProtSSN-k{k}_h{h} and set --gnn_model_path or download from HuggingFace tyang816/ProtSSN."
        )
    gnn_module.load_state_dict(torch.load(ckpt, map_location=device))
    norm_file = f"src/mutation/models/egnn/norm/cath_k{k}_mean_attr.pt"
    protssn = ProtSSN(
        c_alpha_max_neighbors=k,
        pre_transform=NormalizeProtein(filename=norm_file),
        plm_model=plm_module,
        gnn_model=gnn_module,
    )
    wrapper = ProtSSNEmbeddingWrapper(protssn, device=device)
    # ProtSSN embedding dim = ESM output (1280); "h512" in name is GNN internal hidden_channels, not final dim
    protssn_hidden_size = 1280
    wrapper.hidden_size = protssn_hidden_size
    wrapper.register_buffer("_hidden_size", torch.tensor(protssn_hidden_size, dtype=torch.long))
    return tokenizer, wrapper


def get_hidden_size(plm_model, model_type):
    """Get hidden size based on model type."""
    if "esm" in model_type:
        return plm_model.config.hidden_size
    elif "bert" in model_type:
        return plm_model.config.hidden_size
    elif "prot_t5" in model_type or "ankh" in model_type:
        return plm_model.config.d_model
    elif "ProSST" in model_type:
        return plm_model.config.hidden_size
    elif "SaProt" in model_type:
        return plm_model.config.hidden_size
    elif "Prime" in model_type:
        return plm_model.config.hidden_size
    elif "deep" in model_type:
        return plm_model.config.hidden_size
    elif "ProtSSN" in model_type:
        # Use buffer _hidden_size (survives prepare), else attribute, else GNN config
        if hasattr(plm_model, "_hidden_size") and plm_model._hidden_size is not None:
            return int(plm_model._hidden_size.item())
        h = getattr(plm_model, "hidden_size", None)
        if h is not None:
            return int(h)
        if hasattr(plm_model, "protssn") and hasattr(plm_model.protssn, "gnn_model"):
            cfg = getattr(plm_model.protssn.gnn_model, "gnn_config", None)
            if cfg and "hidden_channels" in cfg:
                return int(cfg["hidden_channels"])
        return 512
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_vocab_size(plm_model, structure_seq):
    """Get vocabulary size for structure sequences."""
    if 'esm3_structure_seq' in structure_seq:
        return max(plm_model.config.vocab_size, 4100)
    return plm_model.config.vocab_size 
