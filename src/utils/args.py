import argparse
import json
import os
import warnings
from typing import Dict, Any
from datetime import datetime

def parse_args() -> Dict[str, Any]:
    """Parse and validate command line arguments."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate and process arguments
    validate_args(args)
    process_dataset_config(args)
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

def add_dataset_args(parser: argparse.ArgumentParser):
    """Add dataset-related arguments."""
    data_group = parser.add_argument_group('Dataset Configuration')
    data_group.add_argument('--dataset', type=str)
    data_group.add_argument('--dataset_config', type=str)
    data_group.add_argument('--normalize', type=str)
    data_group.add_argument('--num_labels', type=int)
    data_group.add_argument('--problem_type', type=str)
    data_group.add_argument('--pdb_type', type=str)
    data_group.add_argument('--train_file', type=str)
    data_group.add_argument('--valid_file', type=str)
    data_group.add_argument('--test_file', type=str)
    data_group.add_argument('--metrics', type=str)

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
    train_group.add_argument('--monitor_strategy', default='max', type=str, choices=['max', 'min'])
    train_group.add_argument('--training_method', type=str, default='freeze',
                            choices=['full', 'freeze', 'lora', 'ses-adapter'])
    train_group.add_argument('--structure_seq', type=str, default='')

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
    wandb_group.add_argument('--wandb_project', type=str, default='VenusFactory')
    wandb_group.add_argument('--wandb_run_name', type=str)

def validate_args(args: argparse.Namespace):
    """Validate command line arguments."""
    if args.batch_size is None and args.batch_token is None:
        raise ValueError("batch_size or batch_token must be provided")
    
    if args.training_method == 'ses-adapter':
        if args.structure_seq is None:
            raise ValueError("structure_seq must be provided for ses-adapter")
        args.structure_seq = args.structure_seq.split(',')
    else:
        args.structure_seq = []

def process_dataset_config(args: argparse.Namespace):
    """Process dataset configuration file."""
    if not args.dataset_config:
        return
        
    config = json.load(open(args.dataset_config))
    
    # Update args with dataset config values if not already set
    for key in ['dataset', 'pdb_type', 'train_file', 'valid_file', 'test_file',
                'num_labels', 'problem_type', 'monitor', 'metrics', 'normalize']:
        if getattr(args, key) is None and key in config:
            setattr(args, key, config[key])
    
    # Handle metrics specially
    if args.metrics:
        args.metrics = args.metrics.split(',')
        if args.metrics == ['None']:
            args.metrics = ['loss']
            warnings.warn("No metrics provided, using default metrics: loss")

def setup_output_dirs(args: argparse.Namespace):
    """Setup output directories."""
    if args.output_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.output_dir = os.path.join(args.output_root, current_date)
    else:
        args.output_dir = os.path.join(args.output_root, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

def setup_wandb_config(args: argparse.Namespace):
    """Setup wandb configuration."""
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"VenusFactory-{args.dataset}"
        if args.output_model_name is None:
            args.output_model_name = f"{args.wandb_run_name}.pt" 