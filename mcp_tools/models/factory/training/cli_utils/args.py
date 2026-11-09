import argparse
import json
import os
import warnings
from time import localtime, strftime


def parse_args() -> argparse.Namespace:
    """Parse and validate command line arguments."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Process dataset config first (may set structure_seq, pdb_dir from config)
    process_dataset_config(args)
    validate_args(args)
    setup_output_dirs(args)
    setup_wandb_config(args)
    
    return args

def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser with all training arguments."""
    parser = argparse.ArgumentParser()
    
    # Model parameters
    add_model_args(parser)
    
    # Dataset parameters
    add_dataset_args(parser)
    
    # Training parameters
    add_training_args(parser)
    
    # Output parameters
    add_output_args(parser)
    
    # Wandb parameters
    add_wandb_args(parser)
    
    return parser

def add_model_args(parser: argparse.ArgumentParser):
    """Add model-related arguments."""
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--hidden_size', type=int, default=None)
    model_group.add_argument('--num_attention_head', type=int, default=8)
    model_group.add_argument('--attention_probs_dropout', type=float, default=0.1)
    model_group.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D')
    model_group.add_argument('--pooling_method', type=str, default='mean',
                            choices=['mean', 'attention1d', 'light_attention'])
    model_group.add_argument('--pooling_dropout', type=float, default=0.1)
    # ProtSSN (structure-based, freeze only)
    model_group.add_argument('--gnn_config', type=str, default=None,
                             help='Path to GNN config YAML (default: mcp_tools/models/pretrained/mutation/models/egnn/egnn.yaml)')
    model_group.add_argument('--gnn_model_path', type=str, default=None,
                             help='Directory containing ProtSSN GNN weights (e.g. protssn_k10_h512.pt)')
    model_group.add_argument('--c_alpha_max_neighbors', type=int, default=10,
                             help='ProtSSN graph K (c_alpha_max_neighbors)')

def add_dataset_args(parser: argparse.ArgumentParser):
    """Add dataset-related arguments."""
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--dataset', type=str)
    data_group.add_argument('--dataset_config', type=str)
    data_group.add_argument('--normalize', type=str)
    data_group.add_argument('--num_labels', type=int)
    data_group.add_argument('--problem_type', type=str)
    data_group.add_argument('--sequence_column_name', type=str, default='aa_seq')
    data_group.add_argument('--label_column_name', type=str, default='label')
    data_group.add_argument('--pdb_type', type=str)
    data_group.add_argument('--pdb_dir', type=str, default=None,
                            help='PDB directory to generate missing structure_seq columns (stru_token, foldseek_seq, ss8_seq, etc.)')
    data_group.add_argument('--train_file', type=str)
    data_group.add_argument('--valid_file', type=str)
    data_group.add_argument('--test_file', type=str)
    data_group.add_argument('--metrics', type=str)
    data_group.add_argument('--quick_test', action='store_true', default=False,
                            help='Truncate dataset to max_train_samples, max_validation_samples, max_test_samples for quick test')
    data_group.add_argument('--max_train_samples', type=int, default=50)
    data_group.add_argument('--max_validation_samples', type=int, default=50)
    data_group.add_argument('--max_test_samples', type=int, default=50)

def add_training_args(parser: argparse.ArgumentParser):
    """Add training-related arguments."""
    train_group = parser.add_argument_group('Training Configuration')
    train_group.add_argument('--seed', type=int, default=3407)
    train_group.add_argument('--learning_rate', type=float, default=1e-3)
    train_group.add_argument('--scheduler', type=str, choices=['linear', 'cosine', 'step'])
    train_group.add_argument('--warmup_steps', type=int, default=0)
    train_group.add_argument('--num_workers', type=int, default=4)
    train_group.add_argument('--batch_size', type=int)
    train_group.add_argument('--batch_token', type=int)
    train_group.add_argument('--num_epochs', type=int, default=100)
    train_group.add_argument('--max_seq_len', type=int, default=-1)
    train_group.add_argument('--gradient_accumulation_steps', type=int, default=1)
    train_group.add_argument('--max_grad_norm', type=float, default=-1)
    train_group.add_argument('--patience', type=int, default=10)
    train_group.add_argument('--monitor', type=str)
    train_group.add_argument('--monitor_strategy', type=str, choices=['max', 'min'])
    train_group.add_argument('--training_method', type=str, default='freeze',
                           choices=['full', 'freeze', 'lora', 'ses-adapter', 'plm-lora', 'plm-qlora', 'plm-adalora', 'plm-dora', 'plm-ia3'])
    parser.add_argument("--lora_r", type=int, default=8, help="lora r")
    parser.add_argument("--lora_alpha", type=int, default=32, help="lora_alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="lora_dropout")
    parser.add_argument("--feedforward_modules", type=str, default="w0")

    parser.add_argument(
        "--lora_target_modules",
        nargs="+",
        default=["query", "key", "value"],
        help="lora target module",
    )
    train_group.add_argument('--structure_seq', type=str, default='')

    # Transfer Learning / Continue Training parameters
    transfer_group = parser.add_argument_group('Transfer Learning')
    transfer_group.add_argument(
        '--initial_model_path',
        type=str,
        default=None,
        help='Path to pre-trained model checkpoint to start training from (e.g., ckpt/demo/best_model.pt). '
             'Enables continue training from an existing model on new dataset. '
             'The training_method will be loaded from the model config and cannot be changed.'
    )

def add_output_args(parser: argparse.ArgumentParser):
    """Add output-related arguments."""
    output_group = parser.add_argument_group('Output Configuration')
    output_group.add_argument('--output_model_name', type=str)
    output_group.add_argument('--output_root', default="ckpt")
    output_group.add_argument('--output_dir', default=None)

def add_wandb_args(parser: argparse.ArgumentParser):
    """Add wandb-related arguments."""
    wandb_group = parser.add_argument_group('Wandb Configuration')
    wandb_group.add_argument('--wandb', action='store_true')
    wandb_group.add_argument('--wandb_entity', type=str)
    wandb_group.add_argument('--wandb_project', type=str, default='VenusFactory2')
    wandb_group.add_argument('--wandb_run_name', type=str)

def _normalize_list_option(value, name: str) -> list[str]:
    """Normalize comma-separated CLI values and JSON arrays at one boundary."""
    values = value.split(",") if isinstance(value, str) else value
    if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
        raise ValueError(f"{name} must be a comma-separated string or a list of names")
    return [item.strip() for item in values if item.strip()]


def validate_args(args: argparse.Namespace):
    """Validate command line arguments."""
    if args.batch_size is None and args.batch_token is None:
        raise ValueError("batch_size or batch_token must be provided")

    if args.training_method == "ses-adapter":
        channels = [] if args.structure_seq is None else _normalize_list_option(args.structure_seq, "structure_seq")
        if not channels:
            raise ValueError("structure_seq must be provided for ses-adapter")
        args.structure_seq = channels
    else:
        args.structure_seq = []


def process_dataset_config(args: argparse.Namespace):
    """Merge dataset defaults, then normalize metrics exactly once."""
    if args.dataset_config:
        with open(args.dataset_config, encoding="utf-8") as stream:
            config = json.load(stream)

        # Explicit CLI values take precedence over dataset defaults.
        for key in [
            "dataset", "pdb_type", "pdb_dir", "train_file", "valid_file", "test_file",
            "num_labels", "problem_type", "monitor", "monitor_strategy", "metrics",
            "normalize", "gnn_config", "gnn_model_path", "c_alpha_max_neighbors",
        ]:
            if getattr(args, key) is None and key in config:
                setattr(args, key, config[key])

        # Structure channels and table mappings are defined by the dataset preset.
        # Leave channels in their original representation until validate_args.
        if config.get("structure_seq"):
            args.structure_seq = config["structure_seq"]
        for key in ("structure_merge_key", "structure_merge_key_format"):
            if key in config:
                setattr(args, key, config[key])

        for key in ("sequence_column_name", "label_column_name"):
            info = getattr(args, "_column_override_info", {})
            if key in config:
                old_value = getattr(args, key)
                setattr(args, key, config[key])
                info[key] = {"old": old_value, "new": config[key]}
            else:
                info[key] = {"default": getattr(args, key)}
            args._column_override_info = info

    if args.metrics is not None:
        args.metrics = _normalize_list_option(args.metrics, "metrics")
        if args.metrics == ["None"]:
            args.metrics = ["loss"]
            warnings.warn("No metrics provided, using default metrics: loss", stacklevel=2)

def setup_output_dirs(args: argparse.Namespace):
    """Setup output directories."""
    if args.output_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.output_dir = os.path.join(args.output_root, current_date)
    os.makedirs(args.output_dir, exist_ok=True)

def setup_wandb_config(args: argparse.Namespace):
    """Setup wandb configuration."""
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"VenusFactory2-{args.dataset}"
        if args.output_model_name is None:
            args.output_model_name = f"{args.wandb_run_name}.pt" 
