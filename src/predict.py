import sys
import os
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
from transformers import logging
from tools.train.models.inference_loader import load_inference_model
from tools.train.models.pooling import MeanPooling, Attention1dPoolingHead, LightAttentionPoolingHead

# Ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict protein function. Single sequence: use --aa_seq. Batch: use --input_file with --output_dir and --output_file."
    )
    # Model parameters
    parser.add_argument('--eval_method', type=str, default="freeze", choices=["full", "freeze", "ses-adapter", "plm-lora", "plm-qlora", "plm-adalora", "plm-dora", "plm-ia3"], help="Evaluation method")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model")
    parser.add_argument('--plm_model', type=str, required=True, help="Pretrained language model name or path")
    parser.add_argument('--pooling_method', type=str, default="mean", choices=["mean", "attention1d", "light_attention"], help="Pooling method")
    parser.add_argument('--problem_type', type=str, default="single_label_classification",
                        choices=["single_label_classification", "multi_label_classification", "regression",
                                 "residue_single_label_classification", "residue_regression"],
                        help="Problem type")
    parser.add_argument('--num_labels', type=int, default=2, help="Number of labels")
    parser.add_argument('--hidden_size', type=int, default=None, help="Embedding hidden size of the model")
    parser.add_argument('--num_attention_head', type=int, default=8, help="Number of attention heads")
    parser.add_argument('--attention_probs_dropout', type=float, default=0, help="Attention probs dropout prob")
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help="Pooling dropout")
    # Input: single sequence
    parser.add_argument('--aa_seq', type=str, default=None, help="Single amino acid sequence (omit for batch mode)")
    parser.add_argument('--foldseek_seq', type=str, default="", help="Foldseek sequence (optional)")
    parser.add_argument('--ss8_seq', type=str, default="", help="Secondary structure sequence (optional)")
    # Input: batch (CSV)
    parser.add_argument('--input_file', type=str, default=None, help="Input CSV for batch prediction (requires --output_dir, --output_file)")
    parser.add_argument('--output_dir', type=str, default=None, help="Output directory for batch prediction")
    parser.add_argument('--output_file', type=str, default=None, help="Output CSV filename for batch prediction")
    # Other
    parser.add_argument('--dataset', type=str, default="single", help="Dataset name (optional)")
    parser.add_argument('--use_foldseek', action='store_true', help="Use foldseek sequence")
    parser.add_argument('--use_ss8', action='store_true', help="Use secondary structure sequence")
    parser.add_argument('--structure_seq', type=str, default=None, help="Structure sequence types (comma-separated)")
    parser.add_argument('--pdb_dir', type=str, default=None, help="PDB directory (for structure models when input lacks structure columns)")
    parser.add_argument('--max_seq_len', type=int, default=1024, help="Maximum sequence length")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size (for batch mode)")
    args = parser.parse_args()
    # Mode: batch if input_file is set
    if args.input_file:
        if not args.output_dir or not args.output_file:
            parser.error("Batch mode (--input_file) requires --output_dir and --output_file")
    else:
        if not args.aa_seq:
            parser.error("Single mode requires --aa_seq, or use --input_file for batch mode")
    args.use_foldseek = bool(getattr(args, 'foldseek_seq', '')) or args.use_foldseek
    args.use_ss8 = bool(getattr(args, 'ss8_seq', '')) or args.use_ss8
    return args


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
    
    # only for ProSST model
    if "ProSST" in plm_model_name and hasattr(args, 'prosst_stru_token') and args.prosst_stru_token:
        try:
            # process ProSST structure tokens
            if isinstance(args.prosst_stru_token, str):
                seq_clean = args.prosst_stru_token.strip("[]").replace(" ","")
                tokens = list(map(int, seq_clean.split(','))) if seq_clean else []
            elif isinstance(args.prosst_stru_token, (list, tuple)):
                tokens = [int(x) for x in args.prosst_stru_token]
            else:
                tokens = []
                
            # add to data dictionary
            if tokens:
                stru_tokens = torch.tensor([tokens], dtype=torch.long)
                data_dict["aa_seq_stru_tokens"] = stru_tokens
            else:
                # if no structure tokens, use zero padding
                data_dict["aa_seq_stru_tokens"] = torch.zeros_like(aa_inputs["input_ids"], dtype=torch.long)
        except Exception as e:
            print(f"Warning: Failed to process ProSST structure tokens: {e}")
            # use zero padding
            data_dict["aa_seq_stru_tokens"] = torch.zeros_like(aa_inputs["input_ids"], dtype=torch.long)
    
    if args.use_foldseek and foldseek_seq:
        data_dict["foldseek_seq_input_ids"] = foldseek_inputs["input_ids"]
    
    if args.use_ss8 and ss8_seq:
        data_dict["ss8_seq_input_ids"] = ss8_inputs["input_ids"]
    
    print("Processed input sequences with keys:", data_dict.keys())
    return data_dict


def process_sequence(args, tokenizer, plm_model_name, aa_seq, foldseek_seq="", ss8_seq="", prosst_stru_token=None):
    """Process a single sequence (for batch mode). Returns data_dict for one sample."""
    original_aa_seq = (aa_seq or "").strip()
    aa_seq = original_aa_seq
    if not aa_seq:
        raise ValueError("Amino acid sequence is empty")
    foldseek_seq = (foldseek_seq or "").strip()
    ss8_seq = (ss8_seq or "").strip()
    if args.use_foldseek and not foldseek_seq:
        print(f"Warning: Foldseek sequence required but not provided for: {aa_seq[:20]}...")
    if args.use_ss8 and not ss8_seq:
        print(f"Warning: SS8 sequence required but not provided for: {aa_seq[:20]}...")
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
    if args.max_seq_len:
        aa_seq = aa_seq[:args.max_seq_len]
        if args.use_foldseek and foldseek_seq:
            foldseek_seq = foldseek_seq[:args.max_seq_len]
        if args.use_ss8 and ss8_seq:
            ss8_seq = ss8_seq[:args.max_seq_len]
    if 'ankh' in plm_model_name:
        aa_inputs = tokenizer.batch_encode_plus([aa_seq], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
        foldseek_inputs = tokenizer.batch_encode_plus([foldseek_seq], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt") if args.use_foldseek and foldseek_seq else None
        ss8_inputs = tokenizer.batch_encode_plus([ss8_seq], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt") if args.use_ss8 and ss8_seq else None
    else:
        aa_inputs = tokenizer([aa_seq], return_tensors="pt", padding=True, truncation=True)
        foldseek_inputs = tokenizer([foldseek_seq], return_tensors="pt", padding=True, truncation=True) if args.use_foldseek and foldseek_seq else None
        ss8_inputs = tokenizer([ss8_seq], return_tensors="pt", padding=True, truncation=True) if args.use_ss8 and ss8_seq else None
    data_dict = {
        "aa_seq_input_ids": aa_inputs["input_ids"],
        "aa_seq_attention_mask": aa_inputs["attention_mask"],
    }
    if "ProSST" in plm_model_name and prosst_stru_token is not None:
        try:
            if isinstance(prosst_stru_token, str):
                tokens = list(map(int, prosst_stru_token.strip("[]").replace(" ", "").split(","))) if prosst_stru_token.strip("[]").replace(" ", "") else []
            else:
                tokens = [int(x) for x in prosst_stru_token]
            data_dict["aa_seq_stru_tokens"] = torch.tensor([tokens], dtype=torch.long) if tokens else torch.zeros_like(aa_inputs["input_ids"], dtype=torch.long)
        except Exception:
            data_dict["aa_seq_stru_tokens"] = torch.zeros_like(aa_inputs["input_ids"], dtype=torch.long)
    if args.use_foldseek and foldseek_seq and foldseek_inputs is not None:
        data_dict["foldseek_seq_input_ids"] = foldseek_inputs["input_ids"]
    if args.use_ss8 and ss8_seq and ss8_inputs is not None:
        data_dict["ss8_seq_input_ids"] = ss8_inputs["input_ids"]
    data_dict["original_aa_seq"] = original_aa_seq
    return data_dict


def predict(model, data_dict, device, args, plm_model, verbose=True):
    """Run prediction on the processed input data. Returns dict with batch-compatible keys (lists)."""
    if verbose:
        print("---------- Running Prediction ----------")
    if "ProPrime_650M_OGT" in args.plm_model:
        with torch.no_grad():
            aa_seq = data_dict['aa_seq_input_ids'].to(device)
            attention_mask = data_dict['aa_seq_attention_mask'].to(device)
            plm_model = plm_model.to(device)
            predictions = plm_model(input_ids=aa_seq, attention_mask=attention_mask).predicted_values.item()
            if verbose:
                print(f"Prediction result: {predictions}")
            return {"predictions": predictions}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.to(device)
    with torch.no_grad():
        outputs = model(plm_model, data_dict)
        if args.problem_type == "regression":
            predictions = outputs.squeeze().cpu().numpy()
            pred = predictions.item() if np.isscalar(predictions) or predictions.size == 1 else predictions.tolist()
            if verbose:
                print(f"Prediction result: {pred}")
            return {"predictions": pred}
        elif args.problem_type == "single_label_classification":
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy().tolist()
            class_probs = probabilities.cpu().numpy().tolist()
            if verbose:
                print(f"Predicted class: {predicted_classes[0] if len(predicted_classes) == 1 else predicted_classes}")
                print(f"Class probabilities: {class_probs[0] if len(class_probs) == 1 else class_probs}")
            return {"predicted_classes": predicted_classes, "probabilities": class_probs}
        elif args.problem_type == "multi_label_classification":
            sigmoid_outputs = torch.sigmoid(outputs)
            predictions = (sigmoid_outputs > 0.5).int().cpu().numpy().tolist()
            probabilities = sigmoid_outputs.cpu().numpy().tolist()
            if verbose:
                print(f"Predicted labels: {predictions}")
                print(f"Label probabilities: {probabilities}")
            return {"predictions": predictions, "probabilities": probabilities}
        elif args.problem_type == "residue_single_label_classification":
            probabilities = torch.nn.functional.softmax(outputs, dim=-1)
            predicted_classes = torch.argmax(probabilities, dim=-1).cpu().numpy().tolist()
            class_probs = probabilities.cpu().numpy().tolist()
            aa_seq = list(getattr(args, "aa_seq", "") or data_dict.get("original_aa_seq", ""))
            if verbose:
                print(f"Predicted residue classes: {predicted_classes}")
                print(f"Residue class probabilities: {class_probs}")
            return {"aa_seq": aa_seq, "predicted_classes": predicted_classes, "probabilities": class_probs}
        elif args.problem_type == "residue_regression":
            predictions = outputs.squeeze().cpu().numpy().tolist()
            aa_seq = list(getattr(args, "aa_seq", "") or data_dict.get("original_aa_seq", ""))
            if verbose:
                print(f"Predicted residue values: {predictions}")
            return {"aa_seq": aa_seq, "predictions": predictions}

def _run_batch(args, model, plm_model, tokenizer, device):
    """Batch mode: read CSV, predict each row, write CSV."""
    df = pd.read_csv(args.input_file)
    print(f"Found {len(df)} sequences in {args.input_file}")
    required = ["aa_seq"]
    if args.use_foldseek:
        required.append("foldseek_seq")
    if args.use_ss8:
        required.append("ss8_seq")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input file missing columns: {', '.join(missing)}")
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        try:
            aa_seq = row["aa_seq"]
            foldseek_seq = row.get("foldseek_seq", "") if args.use_foldseek else ""
            ss8_seq = row.get("ss8_seq", "") if args.use_ss8 else ""
            prosst = row.get("prosst_stru_token", None) if "ProSST" in args.plm_model else None
            data_dict = process_sequence(args, tokenizer, args.plm_model, aa_seq, foldseek_seq, ss8_seq, prosst)
            pred_result = predict(model, data_dict, device, args, plm_model, verbose=False)
            result_row = {"aa_seq": aa_seq}
            if "id" in df.columns:
                result_row["id"] = row["id"]
            if args.problem_type == "regression":
                p = pred_result["predictions"]
                result_row["prediction"] = p[0] if isinstance(p, list) and len(p) == 1 else p
            elif args.problem_type == "single_label_classification":
                result_row["predicted_class"] = pred_result["predicted_classes"][0]
                for i, prob in enumerate(pred_result["probabilities"][0]):
                    result_row[f"class_{i}_prob"] = prob
            elif args.problem_type == "multi_label_classification":
                for i, pred in enumerate(pred_result["predictions"][0]):
                    result_row[f"label_{i}"] = pred
                for i, prob in enumerate(pred_result["probabilities"][0]):
                    result_row[f"label_{i}_prob"] = prob
            elif args.problem_type == "residue_single_label_classification":
                result_row["residue_predictions"] = pred_result["predicted_classes"]
                for pos_idx, pos_probs in enumerate(pred_result["probabilities"]):
                    for class_idx, prob in enumerate(pos_probs):
                        result_row[f"pos_{pos_idx}_class_{class_idx}_prob"] = prob
                if "aa_seq" in pred_result:
                    result_row["aa_seq_residues"] = pred_result["aa_seq"]
            elif args.problem_type == "residue_regression":
                result_row["residue_predictions"] = pred_result["predictions"]
                if "aa_seq" in pred_result:
                    result_row["aa_seq_residues"] = pred_result["aa_seq"]
            results.append(result_row)
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            results.append({"aa_seq": row.get("aa_seq", ""), "error": str(e), **({"id": row["id"]} if "id" in df.columns else {})})
    out_path = os.path.join(args.output_dir, args.output_file)
    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Saved {len(results)} results to {out_path}")
    return 0


def main():
    try:
        args = parse_args()
        model, plm_model, tokenizer, device = load_inference_model(args)
        if args.input_file:
            return _run_batch(args, model, plm_model, tokenizer, device)
        data_dict = process_sequences(args, tokenizer, args.plm_model)
        results = predict(model, data_dict, device, args, plm_model, verbose=True)
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
