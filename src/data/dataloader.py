import json
import torch
import datasets
from torch.utils.data import DataLoader
from .collator import Collator
from .batch_sampler import BatchSampler
from .norm import normalize_dataset
from .structure_seq_merge import ensure_structure_columns
from torch.utils.data import Dataset
from typing import Dict, Any, List, Union
import pandas as pd

def prepare_dataloaders(args, tokenizer, logger):
    """Prepare train, validation and test dataloaders."""
    aa_seq_key = args.sequence_column_name
    # Load datasets
    dataset_dict = datasets.load_dataset(args.dataset)
    
    # Truncate dataset to max_train_samples, max_validation_samples, max_test_samples for quick test
    if args.quick_test:
        max_train_samples = getattr(args, 'max_train_samples', None)
        max_validation_samples = getattr(args, 'max_validation_samples', None)
        max_test_samples = getattr(args, 'max_test_samples', None)
        if max_train_samples is not None and max_train_samples > 0 and 'train' in dataset_dict:
            dataset_dict['train'] = dataset_dict['train'].select(range(max_train_samples))
            if logger:
                logger.info(f"Truncated train dataset to {max_train_samples} samples (max_train_samples={max_train_samples}) before structure merge")
        if max_validation_samples is not None and max_validation_samples > 0 and 'validation' in dataset_dict:
            dataset_dict['validation'] = dataset_dict['validation'].select(range(max_validation_samples))
            if logger:
                logger.info(f"Truncated validation dataset to {max_validation_samples} samples (max_validation_samples={max_validation_samples}) before structure merge")
        if max_test_samples is not None and max_test_samples > 0 and 'test' in dataset_dict:
            dataset_dict['test'] = dataset_dict['test'].select(range(max_test_samples))
            if logger:
                logger.info(f"Truncated test dataset to {max_test_samples} samples (max_test_samples={max_test_samples}) before structure merge")
    
    # Ensure structure columns are present
    dataset_dict = ensure_structure_columns(dataset_dict, args, logger)
    train_dataset = dataset_dict['train']
    train_dataset_token_lengths = [len(item[aa_seq_key]) for item in train_dataset]
    val_dataset = dataset_dict['validation']
    val_dataset_token_lengths = [len(item[aa_seq_key]) for item in val_dataset]
    test_dataset = dataset_dict['test']
    test_dataset_token_lengths = [len(item[aa_seq_key]) for item in test_dataset]
    
    if args.normalize is not None:
        train_dataset, val_dataset, test_dataset = normalize_dataset(
            train_dataset, val_dataset, test_dataset, 
            args.normalize, 
            label_column_name=args.label_column_name
        )
    
    # log dataset info
    logger.info("Dataset Statistics:")
    logger.info("------------------------")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"  Number of train samples: {len(train_dataset)}")
    logger.info(f"  Number of validation samples: {len(val_dataset)}")
    logger.info(f"  Number of test samples: {len(test_dataset)}")
    
    # log 3 data points from train_dataset
    logger.info("Sample 3 data points from train dataset:")
    logger.info(f"  Train data point 1: {train_dataset[0]}")
    logger.info(f"  Train data point 2: {train_dataset[1]}")
    logger.info(f"  Train data point 3: {train_dataset[2]}")
    logger.info("------------------------")
    
    collator = Collator(
        tokenizer=tokenizer,
        max_length=args.max_seq_len if args.max_seq_len > 0 else None,
        structure_seq=args.structure_seq,
        problem_type=args.problem_type,
        plm_model=args.plm_model,
        num_labels=args.num_labels,
        sequence_column_name=args.sequence_column_name,
        label_column_name=args.label_column_name
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

