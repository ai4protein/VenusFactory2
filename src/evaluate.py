"""Evaluate Factory sequence models with task-appropriate predictions and metrics."""
import argparse
import json
import os
import re
import sys

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    AUROC,
    Accuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    F1Score,
    MatthewsCorrCoef,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    Precision,
    Recall,
)
from torchmetrics.regression import MeanSquaredError, SpearmanCorrCoef
from tqdm import tqdm

from mcp_tools.models.factory.training.loop.metrics import MultilabelF1Max

_SCORE_METRICS = (BinaryAUROC, BinaryAveragePrecision, MulticlassAUROC,
                  MulticlassAveragePrecision, MultilabelAUROC, MultilabelAveragePrecision)


def build_metrics(names, problem_type, num_labels, device=None):
    """Construct metrics for the declared task, including an empty selection."""
    constructors = {"accuracy": Accuracy, "recall": Recall, "precision": Precision,
                    "f1": F1Score, "mcc": MatthewsCorrCoef, "auroc": AUROC}
    result = {}
    for name in names:
        if problem_type == "regression":
            regression = {"spearman_corr": SpearmanCorrCoef, "mse": MeanSquaredError}
            if name not in regression:
                raise ValueError(f"Invalid regression metric: {name}")
            metric = regression[name]()
        elif name == "f1_max" and problem_type == "multi_label_classification":
            metric = MultilabelF1Max(num_labels=num_labels)
        elif name in constructors:
            if problem_type == "multi_label_classification":
                options = {"task": "multilabel", "num_labels": num_labels}
            elif num_labels <= 2:
                options = {"task": "binary"}
            else:
                options = {"task": "multiclass", "num_classes": num_labels}
            metric = constructors[name](**options)
        else:
            raise ValueError(f"Invalid metric for {problem_type}: {name}")
        result[name] = metric.to(device)
    return result


def evaluation_loss(problem_type, num_labels):
    if problem_type == "regression":
        return nn.MSELoss()
    if problem_type == "multi_label_classification" or num_labels == 1:
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _batch_predictions(logits, labels, problem_type):
    if problem_type == "regression":
        predictions = logits.reshape(-1)
        targets = labels.reshape(-1).float()
        sample_count = int(labels.shape[0]) if labels.ndim else 1
        if predictions.shape != targets.shape or predictions.numel() != sample_count:
            raise ValueError("Regression requires one prediction for each target.")
        return predictions, targets, predictions, None
    if problem_type == "multi_label_classification":
        if logits.shape != labels.shape or logits.ndim != 2:
            raise ValueError("Multilabel predictions and targets must have shape (samples, labels).")
        probabilities = logits.sigmoid()
        return logits, labels.float(), (probabilities > 0.5).long(), probabilities
    if logits.ndim == 1 or logits.shape[-1] == 1:
        values = logits.reshape(-1)
        probabilities = values.sigmoid()
        return values, labels.reshape(-1).float(), (probabilities > 0.5).long(), probabilities
    if logits.ndim != 2:
        raise ValueError("Classification logits must have shape (samples, classes).")
    probabilities = logits.softmax(dim=1)
    return logits, labels.reshape(-1).long(), logits.argmax(dim=1), probabilities


def evaluation_target_transform(config):
    """Only compare checkpoint outputs with raw labels in a known target space."""
    if getattr(config, "problem_type", None) != "regression":
        return None
    transform = getattr(config, "target_transform", None)
    if transform is not None:
        from mcp_tools.models.factory.target_transform import validate_target_transform
        validate_target_transform(transform)
        return transform
    scale = getattr(config, "prediction_output_scale", None)
    if (not isinstance(scale, dict) or scale.get("value_space") != "original_target"
            or getattr(config, "normalize", None) not in (None, "none")):
        raise ValueError("Regression evaluation against original labels requires verified target-transform "
                         "statistics or an explicitly unnormalized checkpoint; this checkpoint output scale is unknown or normalized.")
    return None


@torch.inference_mode()
def evaluate(model, plm_model, metrics, dataloader, loss_function, device=None, *,
             problem_type=None, label_column_name=None, target_transform=None):
    """Return metric columns and predictions without mutating the metric mapping."""
    legacy_args = globals().get("args")
    problem_type = problem_type or getattr(legacy_args, "problem_type", None)
    if problem_type is None:
        problem_type = "regression" if isinstance(loss_function, (nn.MSELoss, nn.L1Loss)) else "single_label_classification"
    if problem_type not in {"regression", "single_label_classification", "multi_label_classification"}:
        raise ValueError(f"Unsupported evaluation problem type: {problem_type}")
    label_column_name = label_column_name or getattr(legacy_args, "label_column_name", None) or "label"
    if problem_type == "regression" and target_transform is None and legacy_args is not None:
        target_transform = evaluation_target_transform(legacy_args)
    if target_transform is not None and problem_type != "regression":
        raise ValueError("Target transforms apply only to regression evaluation.")
    model.eval()
    if plm_model is not None:
        plm_model.eval()
    for metric in metrics.values():
        metric.reset()
    total_loss, total_samples = 0.0, 0
    pred_labels = []
    epoch_iterator = tqdm(dataloader)
    for batch in epoch_iterator:
        batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
        label = batch[label_column_name]
        logits = model(plm_model, batch)
        if target_transform is not None:
            from mcp_tools.models.factory.target_transform import inverse_targets
            restored = inverse_targets(logits.detach().cpu().numpy(), target_transform)
            logits = torch.as_tensor(restored, dtype=logits.dtype, device=logits.device)
            if not torch.isfinite(logits).all():
                raise ValueError("Inverse target transform exceeds the evaluation tensor range.")
        loss_inputs, loss_targets, predictions, probabilities = _batch_predictions(logits, label, problem_type)
        count = int(label.shape[0]) if label.ndim else 1
        loss = loss_function(loss_inputs, loss_targets)
        mean_loss = loss.mean()
        total_loss += mean_loss.item() if getattr(loss_function, "reduction", "mean") == "sum" else mean_loss.item() * count
        total_samples += count
        pred_labels.extend(predictions.detach().cpu().tolist())
        for metric in metrics.values():
            values = predictions
            if isinstance(metric, _SCORE_METRICS):
                values = probabilities
                if isinstance(metric, (BinaryAUROC, BinaryAveragePrecision)) and values.ndim == 2:
                    values = values[:, 1]
            targets = loss_targets if problem_type == "regression" else label.long().reshape(predictions.shape)
            metric.update(values, targets)
        epoch_iterator.set_postfix(eval_loss=mean_loss.item())
    if not total_samples:
        raise ValueError("The evaluation dataset contains no samples.")
    result = {name: [metric.compute().detach().cpu().tolist()] for name, metric in metrics.items()}
    result["loss"] = [total_loss / total_samples]
    return result, pred_labels


def save_evaluation_results(metrics, predictions, dataframe, directory):
    """Write predictions in input row order using the existing CSV filenames."""
    if len(dataframe) != len(predictions):
        raise ValueError("Prediction count differs from input rows; evaluation must not drop samples.")
    result = dataframe.copy()
    result["pred_label"] = predictions
    os.makedirs(directory, exist_ok=True)
    pd.DataFrame(metrics).to_csv(os.path.join(directory, "evaluation_metrics.csv"), index=False)
    result.to_csv(os.path.join(directory, "evaluation_result.csv"), index=False)


if __name__ == '__main__':
    from datasets import load_dataset
    from transformers import logging

    from mcp_tools.models.factory.training.data.batch_sampler import BatchSampler
    from mcp_tools.models.factory.training.models.inference_loader import load_inference_model

    logging.set_verbosity_error()
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--eval_method', type=str, default=None, help='evaluation method')
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--num_labels', type=int, default=2, help='number of labels')
    parser.add_argument('--pooling_method', type=str, default='mean', help='pooling method')
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help='pooling dropout')
    
    # dataset
    parser.add_argument('--dataset_config', type=str, default=None, help='dataset config json path')
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--problem_type', type=str, default=None, help='problem type')
    parser.add_argument('--sequence_column_name', type=str, default=None, help='sequence column name')
    parser.add_argument('--label_column_name', type=str, default=None, help='label column name')
    parser.add_argument('--test_file', type=str, default=None, help='test file')
    parser.add_argument('--split', type=str, default=None, help='split name in Huggingface')
    parser.add_argument('--test_result_dir', type=str, default=None, help='test result directory')
    parser.add_argument('--metrics', type=str, default=None, help='computation metrics')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size for fixed batch size')
    parser.add_argument('--batch_token', type=int, default=10000, help='max number of token per batch')
    parser.add_argument('--use_foldseek', action='store_true', help='use foldseek')
    parser.add_argument('--use_ss8', action='store_true', help='use ss8')
    
    # model path
    parser.add_argument('--output_model_name', type=str, default=None, help='model name')
    parser.add_argument('--output_root', default="result", help='root directory to save trained models')
    parser.add_argument('--output_dir', default=None, help='directory to save trained models')
    parser.add_argument('--model_path', default=None, help='model path directly')
    parser.add_argument('--structure_seq', type=str, default="", help='structure sequence')
    parser.add_argument('--pdb_dir', type=str, default=None, help='PDB directory (for structure models when dataset lacks structure)')
    parser.add_argument('--training_method', type=str, default="freeze", help='training method')
    args = parser.parse_args()

    if args.dataset_config:
        if not os.path.exists(args.dataset_config):
            raise FileNotFoundError(f"dataset_config file not found: {args.dataset_config}")
        with open(args.dataset_config, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        if not args.dataset:
            args.dataset = cfg.get("dataset")
        if not args.test_file:
            args.test_file = cfg.get("test_file") or cfg.get("dataset")
        if not args.split:
            args.split = cfg.get("split")
        if args.problem_type is None:
            args.problem_type = cfg.get("problem_type")
        if args.sequence_column_name is None:
            args.sequence_column_name = cfg.get("sequence_column_name")
        if args.label_column_name is None:
            args.label_column_name = cfg.get("label_column_name")
        if args.metrics is None:
            cfg_metrics = cfg.get("metrics")
            if isinstance(cfg_metrics, list):
                args.metrics = ",".join([str(m).strip() for m in cfg_metrics if str(m).strip()])
            elif isinstance(cfg_metrics, str):
                args.metrics = cfg_metrics
        if "--num_labels" not in sys.argv and cfg.get("num_labels") is not None:
            args.num_labels = cfg.get("num_labels")
    
    if 'foldseek_seq' in args.structure_seq:
        args.use_foldseek = True
        print("Enabled foldseek_seq based on structure_seq parameter")
    if 'ss8_seq' in args.structure_seq:
        args.use_ss8 = True
        print("Enabled ss8_seq based on structure_seq parameter")
    
    if args.test_file is None:
        raise ValueError("test_file is required (can be provided directly or via --dataset_config)")
    if args.test_result_dir is None:
        raise ValueError("test_result_dir is required")

    os.makedirs(args.test_result_dir, exist_ok=True)
    model, plm_model, tokenizer, device = load_inference_model(args)
    target_transform = evaluation_target_transform(args)

    metric_names = [name.strip() for name in (args.metrics or "").split(",") if name.strip()]
    metrics_dict = build_metrics(metric_names, args.problem_type, args.num_labels, device)

    def param_num(model):
        total = sum([param.numel() for param in model.parameters() if param.requires_grad])
        num_M = total/1e6
        if num_M >= 1000:
            return "Number of parameter: %.2fB" % (num_M/1e3)
        else:
            return "Number of parameter: %.2fM" % (num_M)
    print(param_num(model))
    
    def collate_fn(examples):
        aa_seqs, labels = [], []
        if args.use_foldseek:
            foldseek_seqs = []
        if args.use_ss8:
            ss8_seqs = []
        prosst_stru_tokens = [] if "ProSST" in args.plm_model else None
        
        for e in examples:
            # Use the sequence column name from args if available, otherwise default to "aa_seq"
            sequence_column_name = (getattr(args, 'sequence_column_name', None) or 'aa_seq')
            aa_seq = e[sequence_column_name]
            if args.use_foldseek:
                foldseek_seq = e["foldseek_seq"]
            if args.use_ss8:
                ss8_seq = e["ss8_seq"]
            

            if "ProSST" in args.plm_model and "prosst_stru_token" in e:
                stru_token = e["prosst_stru_token"]
                if isinstance(stru_token, str):
                    seq_clean = stru_token.strip("[]").replace(" ","")
                    tokens = list(map(int, seq_clean.split(','))) if seq_clean else []
                elif isinstance(stru_token, (list, tuple)):
                    tokens = [int(x) for x in stru_token]
                else:
                    tokens = []
                prosst_stru_tokens.append(torch.tensor(tokens))
            
            if 'prot_bert' in args.plm_model or "prot_t5" in args.plm_model:
                aa_seq = " ".join(list(aa_seq))
                aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
                if args.use_foldseek:
                    foldseek_seq = " ".join(list(foldseek_seq))
                if args.use_ss8:
                    ss8_seq = " ".join(list(ss8_seq))
            elif 'ankh' in args.plm_model:
                aa_seq = list(aa_seq)
                if args.use_foldseek:
                    foldseek_seq = list(foldseek_seq)
                if args.use_ss8:
                    ss8_seq = list(ss8_seq)
            
            aa_seqs.append(aa_seq)
            if args.use_foldseek:
                foldseek_seqs.append(foldseek_seq)
            if args.use_ss8:
                ss8_seqs.append(ss8_seq)
            # Use the label column name from args if available, otherwise default to "label"
            label_column_name = (getattr(args, 'label_column_name', None) or 'label')
            labels.append(e[label_column_name])
        
        if 'ankh' in args.plm_model:
            aa_inputs = tokenizer.batch_encode_plus(aa_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")
            if args.use_foldseek:
                foldseek_input_ids = tokenizer.batch_encode_plus(foldseek_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")["input_ids"]
            if args.use_ss8:
                ss8_input_ids = tokenizer.batch_encode_plus(ss8_seqs, add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt")["input_ids"]
        else:
            aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
            if args.use_foldseek:
                foldseek_input_ids = tokenizer(foldseek_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
            if args.use_ss8:
                ss8_input_ids = tokenizer(ss8_seqs, return_tensors="pt", padding=True, truncation=True)["input_ids"]
        
        aa_input_ids = aa_inputs["input_ids"]
        attention_mask = aa_inputs["attention_mask"]
        
        if args.problem_type == 'regression':
            labels = torch.as_tensor(labels, dtype=torch.float)
        else:
            labels = torch.as_tensor(labels, dtype=torch.long)
        
        data_dict = {
            "aa_seq_input_ids": aa_input_ids,
            "aa_seq_attention_mask": attention_mask,
            label_column_name: labels
        }
        
        if "ProSST" in args.plm_model and prosst_stru_tokens:
            aa_max_length = len(aa_input_ids[0])
            padded_tokens = []
            for tokens in prosst_stru_tokens:
                if tokens is None or len(tokens) == 0:

                    padded_tokens.append([0] * aa_max_length)
                else:
                    struct_sequence = tokens.tolist()

                    padded_tokens.append(struct_sequence + [0] * (aa_max_length - len(struct_sequence)))
            
            data_dict["aa_seq_stru_tokens"] = torch.tensor(padded_tokens, dtype=torch.long)
        
        if args.use_foldseek:
            data_dict["foldseek_seq_input_ids"] = foldseek_input_ids
        if args.use_ss8:
            data_dict["ss8_seq_input_ids"] = ss8_input_ids
        
        return data_dict
        
    loss_function = evaluation_loss(args.problem_type, args.num_labels)
    
    def process_data_line(data):
        # Use the column names from args if available, otherwise use defaults
        sequence_column_name = (getattr(args, 'sequence_column_name', None) or 'aa_seq')
        label_column_name = (getattr(args, 'label_column_name', None) or 'label')
        
        if args.problem_type == 'multi_label_classification':
            label_list = data[label_column_name].split(',')
            data[label_column_name] = [int(l) for l in label_list]
            binary_list = [0] * args.num_labels
            for index in data[label_column_name]:
                binary_list[index] = 1
            data[label_column_name] = binary_list
        if args.max_seq_len is not None:
            data[sequence_column_name] = data[sequence_column_name][:args.max_seq_len]
            if args.use_foldseek:
                data["foldseek_seq"] = data["foldseek_seq"][:args.max_seq_len]
            if args.use_ss8:
                data["ss8_seq"] = data["ss8_seq"][:args.max_seq_len]
            # 如果是 ProSST 模型且有结构标记，也需要截断
            if "ProSST" in args.plm_model and "prosst_stru_token" in data:
                # 结构标记可能是字符串或列表形式
                if isinstance(data["prosst_stru_token"], str):

                    pass
                elif isinstance(data["prosst_stru_token"], (list, tuple)):
                    data["prosst_stru_token"] = data["prosst_stru_token"][:args.max_seq_len]
            token_num = min(len(data[sequence_column_name]), args.max_seq_len)
        else:
            token_num = len(data[sequence_column_name])
        return data, token_num
    
    # process dataset from json file
    def process_dataset_from_json(file):
        dataset, token_nums = [], []
        for l in open(file):
            data = json.loads(l)
            data, token_num = process_data_line(data)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums


    # process dataset from list
    def process_dataset_from_list(data_list):
        dataset, token_nums = [], []
        for l in data_list:
            data, token_num = process_data_line(l)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums
    
    
    if args.test_file.endswith('json'):
        test_dataset, test_token_num = process_dataset_from_json(args.test_file)
        test_result_df = pd.DataFrame(test_dataset)
    elif args.test_file.endswith('csv'):
        test_dataset, test_token_num = process_dataset_from_list(load_dataset("csv", data_files=args.test_file)['train'])
        if args.test_result_dir:
            test_result_df = pd.read_csv(args.test_file)
    elif '/' in args.test_file:  # Huggingface dataset (only csv now)
        raw_dataset = load_dataset(args.test_file)
        # Using the chosen split first.
        if args.split and args.split in raw_dataset:
            split = args.split
        elif 'test' in raw_dataset:
            split = 'test'
        elif 'validation' in raw_dataset:
            split = 'validation'
        elif 'train' in raw_dataset:
            split = 'train'
        else:
            split = list(raw_dataset.keys())[0]
        
        test_dataset, test_token_num = process_dataset_from_list(raw_dataset[split])
        if args.test_result_dir:
            test_result_df = pd.DataFrame(raw_dataset[split])
    else:
        raise ValueError("Invalid file format")
    
    
    if args.batch_size is None:
        if args.batch_token is None:
            raise ValueError("batch_size or batch_token must be specified")
        if any(count > args.batch_token for count in test_token_num):
            raise ValueError("batch_token is smaller than an input sequence; increase it or use batch_size.")
        test_loader = DataLoader(
            test_dataset, 
            num_workers=args.num_workers, 
            collate_fn=collate_fn,
            batch_sampler=BatchSampler(test_token_num, args.batch_token, False)
        )
    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            shuffle=False
        )

    print("---------- Start Eval ----------")
    with torch.no_grad():
        metric, pred_labels = evaluate(model, plm_model, metrics_dict, test_loader, loss_function, device,
                                       problem_type=args.problem_type, label_column_name=args.label_column_name,
                                       target_transform=target_transform)
        if args.test_result_dir:
            save_evaluation_results(metric, pred_labels, test_result_df, args.test_result_dir)
