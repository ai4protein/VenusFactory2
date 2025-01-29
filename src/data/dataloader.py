import json
import torch
import datasets
from torch.utils.data import DataLoader
from .collator import Collator
from .batch_sampler import BatchSampler
from .norm import min_max_normalize_dataset
from torch.utils.data import Dataset
from typing import Dict, Any, List, Union
import pandas as pd

def prepare_dataloaders(args):
    """Prepare train, validation and test dataloaders."""
    # Process datasets
    train_dataset = ProteinDataset(datasets.load_dataset(args.dataset)['train'], args)
    val_dataset = ProteinDataset(datasets.load_dataset(args.dataset)['validation'], args)
    test_dataset = ProteinDataset(datasets.load_dataset(args.dataset)['test'], args)
    
    if args.normalize == 'min_max':
        train_dataset, val_dataset, test_dataset = min_max_normalize_dataset(train_dataset, val_dataset, test_dataset)
    
    collator = Collator(
        tokenizer=args.tokenizer,
        max_length=args.max_seq_len if args.max_seq_len > 0 else None,
        structure_seq=args.structure_seq,
        problem_type=args.problem_type,
        plm_model=args.plm_model
    )
    
    # Common dataloader parameters
    dataloader_params = {
        'num_workers': args.num_workers,
        'collate_fn': collator
    }
    
    # Create dataloaders based on batching strategy
    if args.batch_token is not None:
        train_loader = create_token_based_loader(train_dataset, args.batch_token, True, **dataloader_params)
        val_loader = create_token_based_loader(val_dataset, args.batch_token, False, **dataloader_params)
        test_loader = create_token_based_loader(test_dataset, args.batch_token, False, **dataloader_params)
    else:
        train_loader = create_size_based_loader(train_dataset, args.batch_size, True, **dataloader_params)
        val_loader = create_size_based_loader(val_dataset, args.batch_size, False, **dataloader_params)
        test_loader = create_size_based_loader(test_dataset, args.batch_size, False, **dataloader_params)
    
    return train_loader, val_loader, test_loader

def create_token_based_loader(dataset, batch_token, shuffle, **kwargs):
    """Create dataloader with token-based batching."""
    sampler = BatchSampler(dataset.token_lengths, batch_token, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)

def create_size_based_loader(dataset, batch_size, shuffle, **kwargs):
    """Create dataloader with size-based batching."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

class ProteinDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], args):
        self.data = data
        self.args = args
        self.token_lengths = [len(item['aa_seq']) for item in data]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
