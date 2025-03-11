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
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer, AutoModel, AutoModelForMaskedLM
from transformers import logging
from peft import PeftModel

# Import project modules
from models.adapter_model import AdapterModel
from models.lora_model import LoraModel
from models.pooling import MeanPooling, Attention1dPoolingHead, LightAttentionPoolingHead

# Ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Batch predict protein function for multiple sequences")
    
    # Model parameters
    parser.add_argument('--eval_method', type=str, default="freeze", choices=["freeze", "plm-lora", "plm-qlora", "ses-adapter"], help="Evaluation method")
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
    
    # Input and output parameters
    parser.add_argument('--input_file', type=str, required=True, help="Path to input CSV file with sequences")
    parser.add_argument('--output_file', type=str, required=True, help="Path to output CSV file for predictions")
    parser.add_argument('--use_foldseek', action='store_true', help="Use foldseek sequence")
    parser.add_argument('--use_ss8', action='store_true', help="Use secondary structure sequence")
    parser.add_argument('--structure_seq', type=str, default=None, help="Structure sequence types to use (comma-separated)")
    
    # Other parameters
    parser.add_argument('--max_seq_len', type=int, default=1024, help="Maximum sequence length")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for prediction")
    parser.add_argument('--dataset', type=str, default="Protein-wise", help="Dataset name")
    
    args = parser.parse_args()
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
    elif "ProSST" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = AutoModelForMaskedLM.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
    elif "Prime" in args.plm_model:
        tokenizer = AutoTokenizer.from_pretrained(args.plm_model, do_lower_case=False)
        plm_model = AutoModelForMaskedLM.from_pretrained(args.plm_model).to(device).eval()
        args.hidden_size = plm_model.config.hidden_size
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
    
    print(f"Training method: {args.eval_method}")  # Default for prediction
    print(f"Structure sequence: {args.structure_seq}")
    print(f"Use foldseek: {args.use_foldseek}")
    print(f"Use ss8: {args.use_ss8}")
    print(f"Problem type: {args.problem_type}")
    print(f"Number of labels: {args.num_labels}")
    print(f"Number of attention heads: {args.num_attention_head}")
    
    # Create and load model
    try:
        if args.eval_method in ["ses-adapter", "freeze"]:
            model = AdapterModel(args)
        # ! lora/ qlora
        elif args.eval_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']:
            model = LoraModel(args)
        if args.model_path is not None:
            model_path = args.model_path
        else:
            model_path = f"{args.output_root}/{args.output_dir}/{args.output_model_name}"
        model.load_state_dict(torch.load(model_path))
        model.to(device).eval()
        # ! lora/ qlora
        if args.eval_method == 'plm-lora':
            lora_path = model_path.replace(".pt", "_lora.pt")
            plm_model = PeftModel.from_pretrained(plm_model,lora_path)
            plm_model = plm_model.merge_and_unload()
        elif args.eval_method == 'plm-qlora':
            lora_path = model_path.replace(".pt", "_qlora.pt")
            plm_model = PeftModel.from_pretrained(plm_model,lora_path)
            plm_model = plm_model.merge_and_unload()
        elif args.eval_method == "plm-dora":
            dora_path = model_path.replace(".pt", "_dora.pt")
            plm_model = PeftModel.from_pretrained(plm_model, dora_path)
            plm_model = plm_model.merge_and_unload()
        elif args.eval_method == "plm-adalora":
            adalora_path = model_path.replace(".pt", "_adalora.pt")
            plm_model = PeftModel.from_pretrained(plm_model, adalora_path)
            plm_model = plm_model.merge_and_unload()
        elif args.eval_method == "plm-ia3":
            ia3_path = model_path.replace(".pt", "_ia3.pt")
            plm_model = PeftModel.from_pretrained(plm_model, ia3_path)
            plm_model = plm_model.merge_and_unload()
            plm_model.to(device).eval()  
        return model, plm_model, tokenizer, device
    
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

def process_sequence(args, tokenizer, plm_model_name, aa_seq, foldseek_seq="", ss8_seq="", prosst_stru_token=None):
    """Process and prepare a single input sequence for prediction"""
    
    # Process amino acid sequence
    aa_seq = aa_seq.strip()
    if not aa_seq:
        raise ValueError("Amino acid sequence is empty")
    
    # Process structure sequences if needed
    foldseek_seq = foldseek_seq.strip() if foldseek_seq else ""
    ss8_seq = ss8_seq.strip() if ss8_seq else ""
    
    # Check if structure sequences are required but not provided
    if args.use_foldseek and not foldseek_seq:
        print(f"Warning: Foldseek sequence is required but not provided for sequence: {aa_seq[:20]}...")
    if args.use_ss8 and not ss8_seq:
        print(f"Warning: SS8 sequence is required but not provided for sequence: {aa_seq[:20]}...")
    
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
    
    if "ProSST" in plm_model_name and prosst_stru_token is not None:
        try:
            if isinstance(prosst_stru_token, str):
                seq_clean = prosst_stru_token.strip("[]").replace(" ","")
                tokens = list(map(int, seq_clean.split(','))) if seq_clean else []
            elif isinstance(prosst_stru_token, (list, tuple)):
                tokens = [int(x) for x in prosst_stru_token]
            else:
                tokens = []
                
            if tokens:
                stru_tokens = torch.tensor([tokens], dtype=torch.long)
                data_dict["aa_seq_stru_tokens"] = stru_tokens
            else:
                data_dict["aa_seq_stru_tokens"] = torch.zeros_like(aa_inputs["input_ids"], dtype=torch.long)
        except Exception as e:
            print(f"Warning: Failed to process ProSST structure tokens: {e}")
            data_dict["aa_seq_stru_tokens"] = torch.zeros_like(aa_inputs["input_ids"], dtype=torch.long)
    
    if args.use_foldseek and foldseek_seq:
        data_dict["foldseek_seq_input_ids"] = foldseek_inputs["input_ids"]
    
    if args.use_ss8 and ss8_seq:
        data_dict["ss8_seq_input_ids"] = ss8_inputs["input_ids"]
    
    return data_dict

def predict_batch(model, plm_model, data_dict, device, args):
    """Run prediction on a batch of processed input data"""
    
    # Move data to device
    for k, v in data_dict.items():
        data_dict[k] = v.to(device)
    
    # Run model inference
    with torch.no_grad():
        outputs = model(plm_model, data_dict)
        
        # Process outputs based on problem type
        if args.problem_type == "regression":
            predictions = outputs.squeeze().cpu().numpy()
            return {"predictions": predictions.tolist()}
        
        elif args.problem_type == "single_label_classification":
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            class_probs = probabilities.cpu().numpy()
            
            return {
                "predicted_classes": predicted_classes.tolist(),
                "probabilities": class_probs.tolist()
            }
        
        elif args.problem_type == "multi_label_classification":
            sigmoid_outputs = torch.sigmoid(outputs)
            predictions = (sigmoid_outputs > 0.5).int().cpu().numpy()
            probabilities = sigmoid_outputs.cpu().numpy()
            
            return {
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist()
            }

def main():
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Load model and tokenizer
        model, plm_model, tokenizer, device = load_model_and_tokenizer(args)
        
        # Read input CSV file
        print(f"---------- Reading input file: {args.input_file} ----------")
        try:
            df = pd.read_csv(args.input_file)
            print(f"Found {len(df)} sequences in input file")
        except Exception as e:
            print(f"Error reading input file: {str(e)}")
            sys.exit(1)
        
        # Check required columns
        required_columns = ["aa_seq"]
        if args.use_foldseek:
            required_columns.append("foldseek_seq")
        if args.use_ss8:
            required_columns.append("ss8_seq")
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Error: Input file is missing required columns: {', '.join(missing_columns)}")
            sys.exit(1)
        
        # Initialize results dataframe
        results = []
        
        # Process each sequence
        print("---------- Processing sequences ----------")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            try:
                # Get sequences from row
                aa_seq = row["aa_seq"]
                foldseek_seq = row["foldseek_seq"] if "foldseek_seq" in df.columns and args.use_foldseek else ""
                ss8_seq = row["ss8_seq"] if "ss8_seq" in df.columns and args.use_ss8 else ""
                
                # Process sequence
                data_dict = process_sequence(args, tokenizer, args.plm_model, aa_seq, foldseek_seq, ss8_seq)
                
                # Run prediction
                prediction_results = predict_batch(model, plm_model, data_dict, device, args)
                
                # Create result row
                result_row = {"aa_seq": aa_seq}
                
                # Add sequence ID if available
                if "id" in df.columns:
                    result_row["id"] = row["id"]
                
                # Add prediction results based on problem type
                if args.problem_type == "regression":
                    result_row["prediction"] = prediction_results["predictions"][0]
                
                elif args.problem_type == "single_label_classification":
                    result_row["predicted_class"] = prediction_results["predicted_classes"][0]
                    
                    # Add class probabilities
                    for i, prob in enumerate(prediction_results["probabilities"][0]):
                        result_row[f"class_{i}_prob"] = prob
                
                elif args.problem_type == "multi_label_classification":
                    # Add binary predictions
                    for i, pred in enumerate(prediction_results["predictions"][0]):
                        result_row[f"label_{i}"] = pred
                    
                    # Add probabilities
                    for i, prob in enumerate(prediction_results["probabilities"][0]):
                        result_row[f"label_{i}_prob"] = prob
                
                results.append(result_row)
                
            except Exception as e:
                print(f"Error processing sequence at index {idx}: {str(e)}")
                # Add error row
                error_row = {"aa_seq": aa_seq, "error": str(e)}
                if "id" in df.columns:
                    error_row["id"] = row["id"]
                results.append(error_row)
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        
        # Save results to output file
        print(f"---------- Saving results to {args.output_file} ----------")
        results_df.to_csv(args.output_file, index=False)
        print(f"Saved {len(results_df)} prediction results")
        
        print("---------- Batch prediction completed successfully ----------")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 