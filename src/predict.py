#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import argparse
import torch
import re
import json
import os
import warnings
import numpy as np
from pathlib import Path
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel
from transformers import logging

# Import project modules
from models.adapter_model import AdapterModel
from models.pooling import MeanPooling, Attention1dPoolingHead, LightAttentionPoolingHead

# Ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Predict protein function for a single sequence")
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--plm_model', type=str, required=True, help="Pretrained language model name or path")
    parser.add_argument('--pooling_method', type=str, default="mean", choices=["mean", "attention1d", "light_attention"], help="Pooling method")
    parser.add_argument('--problem_type', type=str, default="single_label_classification", 
                        choices=["single_label_classification", "multi_label_classification", "regression"], 
                        help="Problem type")
    parser.add_argument('--num_labels', type=int, default=2, help="Number of labels")
    parser.add_argument('--hidden_size', type=int, default=None, help="Embedding hidden size of the model")
    parser.add_argument('--num_attention_head', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--attention_probs_dropout', type=float, default=0, help="Attention probs dropout prob")
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help="Pooling dropout")
    
    # Input sequence parameters
    parser.add_argument('--aa_seq', type=str, required=True, help="Amino acid sequence")
    parser.add_argument('--foldseek_seq', type=str, default="", help="Foldseek sequence (optional)")
    parser.add_argument('--ss8_seq', type=str, default="", help="Secondary structure sequence (optional)")
    parser.add_argument('--dataset', type=str, default="single", help="Dataset name (optional)")
    parser.add_argument('--use_foldseek', action='store_true', help="Use foldseek sequence")
    parser.add_argument('--use_ss8', action='store_true', help="Use secondary structure sequence")
    parser.add_argument('--structure_seq', type=str, default=None, help="Structure sequence types to use (comma-separated)")
    
    # Other parameters
    parser.add_argument('--max_seq_len', type=int, default=1024, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Automatically determine whether to use structure sequences based on input
    args.use_foldseek = bool(args.foldseek_seq)
    args.use_ss8 = bool(args.ss8_seq)
    
    return args

def load_model_and_tokenizer(args):
    print("---------- Loading Model and Tokenizer ----------")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Load model configuration if available
    config_path = os.path.join(os.path.dirname(args.model_path), "config.json")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            print(f"Loaded configuration from {config_path}")
            
            # Update args with config values if they exist
            if "pooling_method" in config:
                args.pooling_method = config["pooling_method"]
            if "problem_type" in config:
                args.problem_type = config["problem_type"]
            if "num_labels" in config:
                args.num_labels = config["num_labels"]
            if "num_attention_head" in config:
                args.num_attention_head = config["num_attention_head"]
            if "attention_probs_dropout" in config:
                args.attention_probs_dropout = config["attention_probs_dropout"]
            if "pooling_dropout" in config:
                args.pooling_dropout = config["pooling_dropout"]
    except FileNotFoundError:
        print(f"Model config not found at {config_path}. Using command line arguments.")
    
    # Build tokenizer and protein language model
    if "esm" in args.plm_model:
        tokenizer = EsmTokenizer.from_pretrained(args.plm_model)
        plm_model = EsmModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "bert" in args.plm_model:
        tokenizer = BertTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = BertModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "prot_t5" in args.plm_model:
        tokenizer = T5Tokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    elif "ankh" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = T5EncoderModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.d_model
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model)
        plm_model = AutoModel.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    
    args.vocab_size = plm_model.config.vocab_size
    
    # Determine structure sequence types
    if args.structure_seq is None:
        args.structure_seq = ""
        print("Warning: structure_seq was None, setting to empty string")
    
    # Auto-set structure sequence flags based on structure_seq parameter
    if 'foldseek_seq' in args.structure_seq:
        args.use_foldseek = True
        print("Enabled foldseek_seq based on structure_seq parameter")
    if 'ss8_seq' in args.structure_seq:
        args.use_ss8 = True
        print("Enabled ss8_seq based on structure_seq parameter")
    
    # If flags are set but structure_seq is not, update structure_seq
    structure_seq_list = []
    if args.use_foldseek and 'foldseek_seq' not in args.structure_seq:
        structure_seq_list.append("foldseek_seq")
    if args.use_ss8 and 'ss8_seq' not in args.structure_seq:
        structure_seq_list.append("ss8_seq")
    
    if structure_seq_list and not args.structure_seq:
        args.structure_seq = ",".join(structure_seq_list)
    
    print(f"Training method: freeze")  # Default for prediction
    print(f"Structure sequence: {args.structure_seq}")
    print(f"Use foldseek: {args.use_foldseek}")
    print(f"Use ss8: {args.use_ss8}")
    print(f"Problem type: {args.problem_type}")
    print(f"Number of labels: {args.num_labels}")
    print(f"Number of attention heads: {args.num_attention_head}")
    
    # Create and load model
    try:
        model = AdapterModel(args)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model.to(device).eval()
        return model, plm_model, tokenizer, device
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def process_sequences(args, tokenizer, plm_model_name):
    """Process and prepare input sequences for prediction"""
    print("---------- Processing Input Sequences ----------")
    
    # Process amino acid sequence
    aa_seq = args.aa_seq.strip()
    if not aa_seq:
        raise ValueError("Amino acid sequence is empty")
    
    # Process structure sequences if needed
    foldseek_seq = args.foldseek_seq.strip() if args.foldseek_seq else ""
    ss8_seq = args.ss8_seq.strip() if args.ss8_seq else ""
    
    # Check if structure sequences are required but not provided
    if args.use_foldseek and not foldseek_seq:
        print("Warning: Foldseek sequence is required but not provided.")
    if args.use_ss8 and not ss8_seq:
        print("Warning: SS8 sequence is required but not provided.")
    
    # Format sequences based on model type
    if 'prot_bert' in plm_model_name or "prot_t5" in plm_model_name:
        aa_seq = " ".join(list(aa_seq))
        aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
        if args.use_foldseek and foldseek_seq:
            foldseek_seq = " ".join(list(foldseek_seq))
        if args.use_ss8 and ss8_seq:
            ss8_seq = " ".join(list(ss8_seq))
    elif 'ankh' in plm_model_name:
        aa_seq = list(aa_seq)
        if args.use_foldseek and foldseek_seq:
            foldseek_seq = list(foldseek_seq)
        if args.use_ss8 and ss8_seq:
            ss8_seq = list(ss8_seq)
    
    # Truncate sequences if needed
    if args.max_seq_len:
        aa_seq = aa_seq[:args.max_seq_len]
        if args.use_foldseek and foldseek_seq:
            foldseek_seq = foldseek_seq[:args.max_seq_len]
        if args.use_ss8 and ss8_seq:
            ss8_seq = ss8_seq[:args.max_seq_len]
    
    # Tokenize sequences
    if 'ankh' in plm_model_name:
        aa_inputs = tokenizer.batch_encode_plus([aa_seq], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        if args.use_foldseek and foldseek_seq:
            foldseek_inputs = tokenizer.batch_encode_plus([foldseek_seq], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        if args.use_ss8 and ss8_seq:
            ss8_inputs = tokenizer.batch_encode_plus([ss8_seq], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
    else:
        aa_inputs = tokenizer([aa_seq], return_tensors="pt", padding=True, truncation=True)
        if args.use_foldseek and foldseek_seq:
            foldseek_inputs = tokenizer([foldseek_seq], return_tensors="pt", padding=True, truncation=True)
        if args.use_ss8 and ss8_seq:
            ss8_inputs = tokenizer([ss8_seq], return_tensors="pt", padding=True, truncation=True)
    
    # Prepare data dictionary
    data_dict = {
        "aa_seq_input_ids": aa_inputs["input_ids"],
        "aa_seq_attention_mask": aa_inputs["attention_mask"],
    }
    
    if args.use_foldseek and foldseek_seq:
        data_dict["foldseek_seq_input_ids"] = foldseek_inputs["input_ids"]
    
    if args.use_ss8 and ss8_seq:
        data_dict["ss8_seq_input_ids"] = ss8_inputs["input_ids"]
    
    print("Processed input sequences with keys:", data_dict.keys())
    return data_dict

def predict(model, data_dict, device, args, plm_model):
    """Run prediction on the processed input data"""
    print("---------- Running Prediction ----------")
    
    # Move data to device
    for k, v in data_dict.items():
        data_dict[k] = v.to(device)
    
    # Run model inference
    with torch.no_grad():
        outputs = model(plm_model, data_dict)  # Pass the actual plm_model instead of None
        
        # Process outputs based on problem type
        if args.problem_type == "regression":
            predictions = outputs.squeeze().item()
            print(f"Prediction result: {predictions}")
            return {"prediction": predictions}
        
        elif args.problem_type == "single_label_classification":
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            class_probs = probabilities.squeeze().tolist()
            
            # Ensure class_probs is a list
            if not isinstance(class_probs, list):
                class_probs = [class_probs]
            
            print(f"Predicted class: {predicted_class}")
            print(f"Class probabilities: {class_probs}")
            
            return {
                "predicted_class": predicted_class,
                "probabilities": class_probs
            }
        
        elif args.problem_type == "multi_label_classification":
            sigmoid_outputs = torch.sigmoid(outputs)
            predictions = (sigmoid_outputs > 0.5).int().squeeze().tolist()
            probabilities = sigmoid_outputs.squeeze().tolist()
            
            # Ensure predictions and probabilities are lists
            if not isinstance(predictions, list):
                predictions = [predictions]
            if not isinstance(probabilities, list):
                probabilities = [probabilities]
            
            print(f"Predicted labels: {predictions}")
            print(f"Label probabilities: {probabilities}")
            
            return {
                "predictions": predictions,
                "probabilities": probabilities
            }

def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Load model, tokenizer and get device
        model, plm_model, tokenizer, device = load_model_and_tokenizer(args)
        
        # Process input sequences
        data_dict = process_sequences(args, tokenizer, args.plm_model)
        
        # Run prediction
        results = predict(model, data_dict, device, args, plm_model)
        
        # Output results
        print("\n---------- Prediction Results ----------")
        print(json.dumps(results, indent=2))
        
        return 0
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())