import logging
import os
from typing import Dict, Any, Union, Tuple
import torch.nn as nn
from transformers import PreTrainedModel

def setup_logging(args: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration."""
    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Create formatters and handlers
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(args.output_dir, f'{args.output_model_name.split(".")[0]}_training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Log initial info
    logger.info("Starting training with configuration:")
    for key, value in vars(args).items():
        logger.info(f"{key}: {value}")
    
    return logger 

def count_parameters(model: Union[nn.Module, PreTrainedModel]) -> Tuple[int, int]:
    """
    Count total and trainable parameters in model.
    
    Args:
        model: PyTorch model or Hugging Face model
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def format_parameter_count(count: int) -> str:
    """Format parameter count with appropriate unit."""
    if count >= 1e9:
        return f"{count/1e9:.2f}B"
    elif count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    return str(count)

def print_model_parameters(model: nn.Module, plm_model: PreTrainedModel, logger=None):
    """
    Print parameter statistics for both adapter and PLM models.
    
    Args:
        model: Adapter model
        plm_model: Pre-trained language model
        logger: Optional logger for output
    """
    # Count adapter parameters
    adapter_total, adapter_trainable = count_parameters(model)
    
    # Count PLM parameters
    plm_total, plm_trainable = count_parameters(plm_model)
    
    # Prepare output strings
    output = [
        "------------------------",
        "Model Parameters Statistics:",
        "------------------------",
        f"Adapter Model:",
        f"  Total parameters:     {format_parameter_count(adapter_total)}",
        f"  Trainable parameters: {format_parameter_count(adapter_trainable)}",
        f"Pre-trained Model:",
        f"  Total parameters:     {format_parameter_count(plm_total)}",
        f"  Trainable parameters: {format_parameter_count(plm_trainable)}",
        f"Combined:",
        f"  Total parameters:     {format_parameter_count(adapter_total + plm_total)}",
        f"  Trainable parameters: {format_parameter_count(adapter_trainable + plm_trainable)}",
        f"  Trainable percentage: {((adapter_trainable + plm_trainable) / (adapter_total + plm_total)) * 100:.2f}%",
        "------------------------"
    ]
    
    # Print output
    if logger:
        for line in output:
            logger.info(line)
    else:
        for line in output:
            print(line) 