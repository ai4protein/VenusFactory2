import pandas as pd
import torch
import argparse
import os
import sys
sys.path.append(os.getcwd())
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from transformers import EsmModel, EsmConfig, AutoTokenizer
from math import ceil
from src.data.embed.utils import read_multi_fasta


def esm2_embed_single(model_name_or_path, sequence, pooling=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = EsmModel.from_pretrained(model_name_or_path)
    model.cuda()
    model.eval()
    results = tokenizer([sequence], return_tensors="pt", padding=True, max_length=2048, truncation=True)
    input_ids = results["input_ids"].cuda()
    attention_mask = results["attention_mask"].cuda()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    features = outputs.last_hidden_state
    masked_features = features * attention_mask.unsqueeze(2)
    sum_features = torch.sum(masked_features, dim=1)
    if pooling == 'mean':
        pooled_features = sum_features / attention_mask.sum(dim=1, keepdim=True)
    elif pooling == 'max':
        pooled_features = torch.max(masked_features, dim=1)
    elif pooling == 'sum':
        pooled_features = sum_features
    else:
        pooled_features = features
    return pooled_features.detach().cpu().numpy()


def esm2_embed(model_name_or_path, data, batch_size, output_dir, chunk_id, pooling=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = EsmModel.from_pretrained(model_name_or_path)
    model.cuda()
    model.eval()
    
    def collate_fn(batch):
        sequences = [example["sequence"] for example in batch]
        names = [example["header"] for example in batch]
        results = tokenizer(sequences, return_tensors="pt", padding=True, max_length=2048, truncation=True)
        results["name"] = names
        results["sequence"] = sequences
        return results
    
    res_data = {}
    eval_loader = DataLoader(data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=12)
    
    print(f"ESM2 embedding started for {len(data)} sequences")
    print(f"Pooling method: {pooling}")
    
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            features = outputs.last_hidden_state
            masked_features = features * attention_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            if pooling == 'mean':
                pooled_features = sum_features / attention_mask.sum(dim=1, keepdim=True)
            elif pooling == 'max':
                pooled_features = torch.max(masked_features, dim=1)
            elif pooling == 'sum':
                pooled_features = sum_features
            else:
                pooled_features = features
            for name, seq, feature in zip(batch["name"], batch["sequence"], pooled_features):
                res_data[name] = {"sequence": seq, "emebdding": feature.detach().cpu().numpy()}
            torch.cuda.empty_cache()
            
    out_file = os.path.join(output_dir, f"{model_name_or_path.split('/')[-1]}_{pooling}_{chunk_id}.pt")
    torch.save(res_data, out_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='facebook/esm2_t33_650M_UR50D', help='model name or path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--fasta_file_path', type=str, help='fasta file path or directory path')
    parser.add_argument('--pooling', type=str, default=None, help='pooling method', choices=['mean', 'max', 'sum'])
    parser.add_argument('--chunk_num', type=int, default=1, help='chunk number')
    parser.add_argument('--chunk_id', type=int, default=0, help='chunk id')
    parser.add_argument('--output_dir', type=str, default='dataset/database/embed', help='output file path')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    data = read_multi_fasta(args.fasta_file_path)
    chunk_size = ceil(len(data) / args.chunk_num)
    esm2_embed(
        args.model_name_or_path, data[args.chunk_id*chunk_size: (args.chunk_id+1)*chunk_size], 
        args.batch_size, args.output_dir, args.chunk_id,
        args.pooling
    )
    print(f"ESM2 embedding saved to {args.output_dir}")