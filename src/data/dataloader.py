import json
import torch
import datasets
from torch.utils.data import DataLoader
from .collator import Collator
from .batch_sampler import BatchSampler
from .norm import normalize_dataset
from torch.utils.data import Dataset
from typing import Dict, Any, List, Union
import pandas as pd

def prepare_dataloaders(args, tokenizer):
    """Prepare train, validation and test dataloaders."""
    # Process datasets
    train_dataset = datasets.load_dataset(args.dataset)['train']
    train_dataset_token_lengths = [len(item['aa_seq']) for item in train_dataset]
    val_dataset = datasets.load_dataset(args.dataset)['validation']
    val_dataset_token_lengths = [len(item['aa_seq']) for item in val_dataset]
    test_dataset = datasets.load_dataset(args.dataset)['test']
    test_dataset_token_lengths = [len(item['aa_seq']) for item in test_dataset]
    
    if args.normalize is not None:
        train_dataset, val_dataset, test_dataset = normalize_dataset(train_dataset, val_dataset, test_dataset, args.normalize)
    
    collator = Collator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len if args.max_seq_len > 0 else None,
        structure_seq=args.structure_seq,
        problem_type=args.problem_type,
        plm_model=args.plm_model
    )
    
    # Common dataloader parameters
    dataloader_params = {
        'num_workers': args.num_workers,
        'collate_fn': collator,
        'pin_memory': True,
        'persistent_workers': True if args.num_workers > 0 else False,
        'prefetch_factor': 2,
    }
    
    # Create dataloaders based on batching strategy
    if args.batch_token is not None:
        train_loader = create_token_based_loader(train_dataset, train_dataset_token_lengths, args.batch_token, True, **dataloader_params)
        val_loader = create_token_based_loader(val_dataset, val_dataset_token_lengths, args.batch_token, False, **dataloader_params)
        test_loader = create_token_based_loader(test_dataset, test_dataset_token_lengths, args.batch_token, False, **dataloader_params)
    else:
        train_loader = create_size_based_loader(train_dataset, args.batch_size, True, **dataloader_params)
        val_loader = create_size_based_loader(val_dataset, args.batch_size, False, **dataloader_params)
        test_loader = create_size_based_loader(test_dataset, args.batch_size, False, **dataloader_params)
    
    return train_loader, val_loader, test_loader

def create_token_based_loader(dataset, token_lengths, batch_token, shuffle, **kwargs):
    """Create dataloader with token-based batching."""
    sampler = BatchSampler(token_lengths, batch_token, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)

def create_size_based_loader(dataset, batch_size, shuffle, **kwargs):
    """Create dataloader with size-based batching."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

