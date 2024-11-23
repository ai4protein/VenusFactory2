import argparse
import torch
import re
import json
import os
import warnings
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef, BinaryF1Score
from torchmetrics.regression import SpearmanCorrCoef
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer
from transformers import logging
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils.data_utils import BatchSampler
from utils.metrics import MultilabelF1Max
from models.adapter import AdapterModel

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

def evaluate(model, plm_model, metrics, dataloader, loss_fn, device=None):
    total_loss = 0
    epoch_iterator = tqdm(dataloader)
    pred_labels = []
    
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch["label"]
        logits = model(plm_model, batch)
        pred_labels.extend(logits.argmax(dim=1).cpu().numpy())
        
        for metric_name, metric in metrics_dict.items():
            if args.problem_type == 'regression' and args.num_label == 1:
                loss = loss_fn(logits.squeeze(), label.squeeze())
                metric(logits.squeeze(), label.squeeze())
            elif args.problem_type == 'multi_label_classification':
                loss = loss_fn(logits, label.float())
                metric(logits, label)
            else:
                loss = loss_fn(logits, label)
                metric(torch.argmax(logits, 1), label)
                
        total_loss += loss.item() * len(label)
        epoch_iterator.set_postfix(eval_loss=loss.item())
    
    epoch_loss = total_loss / len(dataloader.dataset)
    for k, v in metrics.items():
        metrics[k] = [v.compute().item()]
        print(f"{k}: {metrics[k][0]}")
    metrics['loss'] = [epoch_loss]
    return metrics, pred_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--num_label', type=int, default=2, help='number of labels')
    parser.add_argument('--pooling_method', type=str, default='attention1d', help='pooling method')
    parser.add_argument('--pooling_dropout', type=float, default=0.25, help='pooling dropout')
    
    # dataset
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--problem_type', type=str, default=None, help='problem type')
    parser.add_argument('--test_file', type=str, default=None, help='test file')
    parser.add_argument('--test_result_dir', type=str, default=None, help='test result directory')
    parser.add_argument('--metrics', type=str, default=None, help='computation metrics')
    parser.add_argument('--num_worker', type=int, default=4, help='number of workers')
    parser.add_argument('--max_seq_len', type=int, default=None, help='max sequence length')
    parser.add_argument('--batch_token', type=int, default=10000, help='max number of token per batch')
    parser.add_argument('--use_foldseek', action='store_true', help='use foldseek')
    parser.add_argument('--use_ss8', action='store_true', help='use ss8')
    
    # model path
    parser.add_argument('--model_name', type=str, default=None, help='model name')
    parser.add_argument('--ckpt_root', default="result", help='root directory to save trained models')
    parser.add_argument('--ckpt_dir', default=None, help='directory to save trained models')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.test_result_dir, exist_ok=True)
    # build tokenizer and protein language model
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
    args.vocab_size = plm_model.config.vocab_size
    
    metrics_dict = {}
    args.metrics = args.metrics.split(',')
    for m in args.metrics:
        if m == 'accuracy':
            if args.num_label == 2:
                metrics_dict[m] = BinaryAccuracy()
            else:
                metrics_dict[m] = Accuracy(task="multiclass", num_classes=args.num_label)
        elif m == 'recall':
            if args.num_label == 2:
                metrics_dict[m] = BinaryRecall()
            else:
                metrics_dict[m] = Recall(task="multiclass", num_classes=args.num_label)
        elif m == 'precision':
            if args.num_label == 2:
                metrics_dict[m] = BinaryPrecision()
            else:
                metrics_dict[m] = Precision(task="multiclass", num_classes=args.num_label)
        elif m == 'f1':
            if args.num_label == 2:
                metrics_dict[m] = BinaryF1Score()
            else:
                metrics_dict[m] = F1Score(task="multiclass", num_classes=args.num_label)
        elif m == 'mcc':
            if args.num_label == 2:
                metrics_dict[m] = BinaryMatthewsCorrCoef()
            else:
                metrics_dict[m] = MatthewsCorrCoef(task="multiclass", num_classes=args.num_label)
        elif m == 'auc':
            if args.num_label == 2:
                metrics_dict[m] = BinaryAUROC()
            else:
                metrics_dict[m] = AUROC(task="multiclass", num_classes=args.num_label)
        elif m == 'f1_max':
            metrics_dict[m] = MultilabelF1Max(num_label=args.num_label)
        elif m == 'spearman_corr':
            metrics_dict[m] = SpearmanCorrCoef()
        else:
            raise ValueError(f"Invalid metric: {m}")
    for metric_name, metric in metrics_dict.items():
        metric.to(device)     
    
    
    # load adapter model
    print("---------- Load Model ----------")
    model = AdapterModel(args)
    model_path = f"{args.ckpt_root}/{args.ckpt_dir}/{args.model_name}"
    model.load_state_dict(torch.load(model_path))
    model.to(device).eval()

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
        for e in examples:
            aa_seq = e["aa_seq"]
            if args.use_foldseek:
                foldseek_seq = e["foldseek_seq"]
            if args.use_ss8:
                ss8_seq = e["ss8_seq"]
            
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
            labels.append(e["label"])
        
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
        
        data_dict = {"aa_input_ids": aa_input_ids, "attention_mask": attention_mask, "label": labels}
        if args.use_foldseek:
            data_dict["foldseek_input_ids"] = foldseek_input_ids
        if args.use_ss8:
            data_dict["ss8_input_ids"] = ss8_input_ids
        return data_dict
        
    loss_fn = nn.CrossEntropyLoss()
    
    def process_data_line(data):
        if args.problem_type == 'multi_label_classification':
            label_list = data['label'].split(',')
            data['label'] = [int(l) for l in label_list]
            binary_list = [0] * args.num_label
            for index in data['label']:
                binary_list[index] = 1
            data['label'] = binary_list
        if args.max_seq_len is not None:
            data["aa_seq"] = data["aa_seq"][:args.max_seq_len]
            if args.use_foldseek:
                data["foldseek_seq"] = data["foldseek_seq"][:args.max_seq_len]
            if args.use_ss8:
                data["ss8_seq"] = data["ss8_seq"][:args.max_seq_len]
            token_num = min(len(data["aa_seq"]), args.max_seq_len)
        else:
            token_num = len(data["aa_seq"])
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
    elif args.test_file.endswith('csv'):
        test_dataset, test_token_num = process_dataset_from_list(load_dataset("csv", data_files=args.test_file)['train'])
        if args.test_result_dir:
            test_result_df = pd.read_csv(args.test_file)
    else:
        raise ValueError("Invalid file format")
    
        
    test_loader = DataLoader(
        test_dataset, num_worker=args.num_worker, collate_fn=collate_fn,
        batch_sampler=BatchSampler(test_token_num, args.batch_token, False)
        )

    print("---------- Start Eval ----------")
    with torch.no_grad():
        metric, pred_labels = evaluate(model, plm_model, metrics_dict, test_loader, loss_fn, device)
        if args.test_result_dir:
            pd.DataFrame(metric).to_csv(f"{args.test_result_dir}/test_metrics.csv", index=False)
            test_result_df["pred_label"] = pred_labels
            test_result_df.to_csv(f"{args.test_result_dir}/test_result.csv", index=False)