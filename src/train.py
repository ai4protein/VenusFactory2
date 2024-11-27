import argparse
import warnings
import torch
import os
import sys
sys.path.append(os.getcwd())
import wandb
import random
import json
import re
import math
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import logging
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef, BinaryF1Score
from torchmetrics.regression import SpearmanCorrCoef
from accelerate import Accelerator
from accelerate.utils import set_seed
from time import strftime, localtime
from datasets import load_dataset
from transformers import EsmTokenizer, EsmModel, BertModel, BertTokenizer
from transformers import T5Tokenizer, T5EncoderModel, AutoTokenizer
from utils.data_utils import BatchSampler
from models.adapter import AdapterModel
from utils.metrics import MultilabelF1Max
from utils.loss_function import MultiClassFocalLossWithAlpha
from utils.norm import min_max_normalize_dataset
from data.get_esm3_structure_seq import VQVAE_SPECIAL_TOKENS

# ignore warning information
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


def train(args, model, plm_model, accelerator, metrics_dict, metrics_monitor_strategy_dict,
          train_loader, val_loader, test_loader, loss_function, optimizer, device):
    best_val_loss, best_val_metric_score = float("inf"), -float("inf")
    val_loss_list, val_metric_list = [], []
    path = os.path.join(args.output_dir, args.output_model_name)
    global_steps = 0
    for epoch in range(args.num_epochs):
        print(f"---------- Epoch {epoch} ----------")
        # train
        model.train()
        epoch_train_loss = 0
        epoch_iterator = tqdm(train_loader)
        for batch in epoch_iterator:
            with accelerator.accumulate(model):
                for k, v in batch.items():
                    batch[k] = v.to(device)
                label = batch["label"]
                logits = model(plm_model, batch)
                if args.problem_type == 'regression' and args.num_labels == 1:
                    loss = loss_function(logits.squeeze(), label.squeeze())
                elif args.problem_type == 'multi_label_classification':
                    loss = loss_function(logits, label.float())
                else:
                    loss = loss_function(logits, label)
                epoch_train_loss += loss.item() * len(label)                
                global_steps += 1
                accelerator.backward(loss)
                if args.max_grad_norm != -1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                epoch_iterator.set_postfix(train_loss=loss.item())
                if args.wandb:
                    wandb.log({"train/loss": loss.item(), "train/epoch": epoch}, step=global_steps)
                    
        train_loss = epoch_train_loss / len(train_loader.dataset)
        print(f'EPOCH {epoch} TRAIN loss: {train_loss:.4f}')
        
        # eval every epoch
        model.eval()
        with torch.no_grad():
            val_loss, val_metric_dict = eval_loop(args, model, plm_model, metrics_dict, val_loader, loss_function, device)
            val_metric_score = val_metric_dict[args.monitor]
            val_metric_list.append(val_metric_score)
            val_loss_list.append(val_loss)
            
            if args.wandb:
                val_log = {"valid/loss": val_loss}
                for metric_name, metric_score in val_metric_dict.items():
                    val_log[f"valid/{metric_name}"] = metric_score
                wandb.log(val_log)
            print(f'EPOCH {epoch} VAL loss: {val_loss:.4f} {args.monitor}: {val_metric_score:.4f}')
    
        # Early stopping and model checkpoint saving
        def save_best_model():
            torch.save(model.state_dict(), path)
            print(f'>>> BEST at epoch {epoch}')
            if args.monitor == 'loss':
                print(f'>>> Loss: {best_val_loss:.4f}')
            else:
                print(f'>>> Loss: {val_loss:.4f}, {args.monitor}: {best_val_metric_score:.4f}')
            for metric_name, metric_score in val_metric_dict.items():
                print(f'>>> {metric_name}: {metric_score:.4f}')
            print(f'>>> Model saved to {path}')

        def check_early_stopping(metric_list, get_best_value):
            best_epoch = metric_list.index(get_best_value(metric_list))
            epochs_without_improvement = len(metric_list) - best_epoch
            if epochs_without_improvement > args.patience:
                print(f'>>> Early stopping at epoch {epoch}')
                return True
            return False

        if args.monitor == "loss":
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_model()
            
            if check_early_stopping(val_loss_list, min):
                break

        else:
            monitor_strategy = metrics_monitor_strategy_dict[args.monitor]
            is_better = (lambda x, y: x > y) if monitor_strategy == 'max' else (lambda x, y: x < y)
            get_best = max if monitor_strategy == 'max' else min

            if is_better(val_metric_score, best_val_metric_score):
                best_val_metric_score = val_metric_score
                save_best_model()

            if check_early_stopping(val_metric_list, get_best):
                break

    
    print(f"TESTING: loading from {path}")
    model.load_state_dict(torch.load(path))
    model.eval()
    with torch.no_grad():
        test_loss, test_metric_dict = eval_loop(args, model, plm_model, metrics_dict, test_loader, loss_function, device)
        test_metric_score = test_metric_dict[args.monitor]
        
        if args.wandb:
            test_log = {"test/loss": test_loss}
            for metric_name, metric_score in test_metric_dict.items():
                test_log[f"test/{metric_name}"] = metric_score
            wandb.log(test_log)
        print(f'EPOCH {epoch} TEST loss: {test_loss:.4f} {args.monitor}: {test_metric_score:.4f}')
        for metric_name, metric_score in test_metric_dict.items():
            print(f'>>> {metric_name}: {metric_score:.4f}')


def eval_loop(args, model, plm_model, metrics_dict, dataloader, loss_function, device=None):
    total_loss = 0
    epoch_iterator = tqdm(dataloader)
    
    for batch in epoch_iterator:
        for k, v in batch.items():
            batch[k] = v.to(device)
        label = batch["label"]
        logits = model(plm_model, batch)

        # Calculate loss first, outside of metrics loop
        if args.problem_type == 'regression' and args.num_labels == 1:
            loss = loss_function(logits.squeeze(), label.squeeze())
        elif args.problem_type == 'multi_label_classification':
            loss = loss_function(logits, label.float())
        else:
            loss = loss_function(logits, label)

        # Update metrics if any exist
        for metric_name, metric in metrics_dict.items():
            if args.problem_type == 'regression' and args.num_labels == 1:
                metric(logits.squeeze(), label.squeeze())
            elif args.problem_type == 'multi_label_classification':
                metric(logits, label)
            else:
                metric(torch.argmax(logits, 1), label)

        total_loss += loss.item() * len(label)
        epoch_iterator.set_postfix(eval_loss=loss.item())
    
    metrics_result_dict = {}
    epoch_loss = total_loss / len(dataloader.dataset)
    for metric_name, metric in metrics_dict.items():
        metrics_result_dict[metric_name] = metric.compute().item()
        metric.reset()
    return epoch_loss, metrics_result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model params
    parser.add_argument('--hidden_size', type=int, default=None, help='embedding hidden size of the model')
    parser.add_argument('--num_attention_head', type=int, default=8, help='number of attention heads')
    parser.add_argument('--attention_probs_dropout', type=float, default=0, help='attention probs dropout prob')
    parser.add_argument('--plm_model', type=str, default='facebook/esm2_t33_650M_UR50D', help='esm model name')
    parser.add_argument('--pooling_method', type=str, default='mean', choices=['mean', 'attention1d', 'light_attention'], help='pooling method')
    parser.add_argument('--pooling_dropout', type=float, default=0.1, help='pooling dropout')
    
    # dataset
    parser.add_argument('--dataset', type=str, default=None, help='dataset name')
    parser.add_argument('--dataset_config', type=str, default=None, help='config of dataset')
    parser.add_argument('--num_labels', type=int, default=None, help='number of labels')
    parser.add_argument('--problem_type', type=str, default=None, help='problem type')
    parser.add_argument('--pdb_type', type=str, default=None, help='pdb type')
    parser.add_argument('--train_file', type=str, default=None, help='train file')
    parser.add_argument('--valid_file', type=str, default=None, help='val file')
    parser.add_argument('--test_file', type=str, default=None, help='test file')
    parser.add_argument('--metrics', type=str, default=None, help='computation metrics')
    
    # train model
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--scheduler', type=str, default=None, choices=['linear', 'cosine', 'step'], help='scheduler')
    parser.add_argument('--warmup_steps', type=int, default=0, help='warmup steps')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--batch_token', type=int, default=None, help='max number of token per batch')
    parser.add_argument('--num_epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--max_seq_len', type=int, default=-1, help='max sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=-1, help='max gradient norm')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    parser.add_argument('--monitor', type=str, default=None, help='monitor metric')
    parser.add_argument('--monitor_strategy', type=str, default=None, choices=['max', 'min'], help='monitor strategy')
    parser.add_argument('--training_method', type=str, default='full', choices=['full', 'freeze', 'lora', 'ses-adapter'], help='training method')
    parser.add_argument('--structure_seq', type=str, default=None, help='structure token for ses-adapter')
    parser.add_argument('--loss_function', type=str, default='cross_entropy', choices=['cross_entropy', 'focal_loss'], help='loss function')
    
    # save model
    parser.add_argument('--output_model_name', type=str, default=None, help='model name')
    parser.add_argument('--output_root', default="ckpt", help='root directory to save trained models')
    parser.add_argument('--output_dir', default=None, help='directory to save trained models')
    
    # wandb log
    parser.add_argument('--wandb', action='store_true', help='use wandb to log')
    parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity')
    parser.add_argument('--wandb_project', type=str, default='ProFactory')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    
    args = parser.parse_args()
    
    # check args
    if args.batch_size is None and args.batch_token is None:
        raise ValueError("batch_size or batch_token must be provided")
    
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
        
    dataset_config = json.loads(open(args.dataset_config).read())
    if args.dataset is None:
        args.dataset = dataset_config['dataset']
    if args.pdb_type is None:
        args.pdb_type = dataset_config['pdb_type']
    if args.train_file is None:
        if dataset_config.get('train_file'):
            args.train_file = dataset_config['train_file']
    if args.valid_file is None:
        if dataset_config.get('valid_file'):
            args.valid_file = dataset_config['valid_file']
    if args.test_file is None:
        if dataset_config.get('test_file'):
            args.test_file = dataset_config['test_file']
    if args.num_labels is None:
        args.num_labels = dataset_config['num_labels']
    if args.problem_type is None:
        args.problem_type = dataset_config['problem_type']
    if args.monitor is None:
        args.monitor = dataset_config['monitor']
    if args.metrics is None:
        args.metrics = dataset_config['metrics'].split(',')
        if args.metrics == ['None']:
            args.metrics = ['loss']
            warnings.warn("No metrics provided, using default metrics: loss")

    # Initialize metrics based on problem type
    metrics_dict = {}
    metrics_monitor_strategy_dict = {}
    for metric_name in args.metrics:
        if args.problem_type == 'regression':
            if metric_name == 'spearman':
                metrics_dict[metric_name] = SpearmanCorrCoef().to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
        elif args.problem_type == 'single_label_classification':
            if metric_name == 'accuracy':
                metrics_dict[metric_name] = Accuracy(task='multiclass', num_classes=args.num_labels).to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
            elif metric_name == 'recall':
                metrics_dict[metric_name] = Recall(task='multiclass', num_classes=args.num_labels).to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
            elif metric_name == 'precision':
                metrics_dict[metric_name] = Precision(task='multiclass', num_classes=args.num_labels).to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
            elif metric_name == 'f1':
                metrics_dict[metric_name] = F1Score(task='multiclass', num_classes=args.num_labels).to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
            elif metric_name == 'mcc':
                metrics_dict[metric_name] = MatthewsCorrCoef(task='multiclass', num_classes=args.num_labels).to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
            elif metric_name == 'auroc':
                metrics_dict[metric_name] = AUROC(task='multiclass', num_classes=args.num_labels).to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
        elif args.problem_type == 'binary_classification':
            if metric_name == 'accuracy':
                metrics_dict[metric_name] = BinaryAccuracy().to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
            elif metric_name == 'recall':
                metrics_dict[metric_name] = BinaryRecall().to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
            elif metric_name == 'precision':
                metrics_dict[metric_name] = BinaryPrecision().to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
            elif metric_name == 'f1':
                metrics_dict[metric_name] = BinaryF1Score().to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
            elif metric_name == 'mcc':
                metrics_dict[metric_name] = BinaryMatthewsCorrCoef().to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
            elif metric_name == 'auroc':
                metrics_dict[metric_name] = BinaryAUROC().to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
        elif args.problem_type == 'multi_label_classification':
            if metric_name == 'f1_max':
                metrics_dict[metric_name] = MultilabelF1Max().to(device)
                metrics_monitor_strategy_dict[metric_name] = 'max'
    
    # Add loss to metrics if it's the monitor metric
    if args.monitor == 'loss':
        metrics_dict['loss'] = 'loss'
        metrics_monitor_strategy_dict['loss'] = 'min'
        
    
    # create checkpoint directory
    if args.output_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.output_dir = os.path.join(args.output_root, current_date)
    else:
        args.output_dir = os.path.join(args.output_root, args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # init wandb
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"ProFactory-{args.dataset}"
        if args.output_model_name is None:
            args.output_model_name = f"{args.wandb_run_name}.pt"
        
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, 
            entity=args.wandb_entity, config=vars(args)
        )
    
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
    
    if args.training_method == 'ses-adapter':
        if args.structure_seq is not None:
            args.structure_seq = args.structure_seq.split(',')
        else:
            raise ValueError("structure_seq must be provided for ses-adapter")
        
        if 'esm3_structure_seq' in args.structure_seq: 
            args.vocab_size = max(plm_model.config.vocab_size, 4100)
        else:
            args.vocab_size = plm_model.config.vocab_size
    else:
        args.structure_seq = []
    
    # load adapter model
    model = AdapterModel(args)
    model.to(device)

    def param_num(model):
        total = sum([param.numel() for param in model.parameters() if param.requires_grad])
        num_M = total/1e6
        if num_M >= 1000:
            return "Number of parameter: %.2fB" % (num_M/1e3)
        else:
            return "Number of parameter: %.2fM" % (num_M)
    print(param_num(model))
    
    def process_data_line(data):
        if args.problem_type == 'multi_label_classification':
            label_list = data['label'].split(',')
            data['label'] = [int(l) for l in label_list]
            binary_list = [0] * args.num_labels
            for index in data['label']:
                binary_list[index] = 1
            data['label'] = binary_list
        
        if 'esm3_structure_seq' in args.structure_seq:
            data["esm3_structure_seq"] = eval(data["esm3_structure_seq"])
        
        if args.max_seq_len > 0:
            data["aa_seq"] = data["aa_seq"][:args.max_seq_len]
            for seq in args.structure_seq:
                data[seq] = data[seq][:args.max_seq_len]
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

    if args.train_file is not None and args.train_file[:-4] == 'json':
        train_dataset, train_token_num = process_dataset_from_json(args.train_file)
        val_dataset, val_token_num = process_dataset_from_json(args.valid_file)
        test_dataset, test_token_num = process_dataset_from_json(args.test_file)
    
    # process dataset from list
    def process_dataset_from_list(data_list):
        dataset, token_nums = [], []
        for l in data_list:
            data, token_num = process_data_line(l)
            dataset.append(data)
            token_nums.append(token_num)
        return dataset, token_nums
    
    if args.train_file == None:
        train_dataset, train_token_num = process_dataset_from_list(load_dataset(args.dataset)['train'])
        val_dataset, val_token_num = process_dataset_from_list(load_dataset(args.dataset)['validation'])
        test_dataset, test_token_num = process_dataset_from_list(load_dataset(args.dataset)['test'])
    
    if dataset_config['normalize'] == 'min_max':
        train_dataset, val_dataset, test_dataset = min_max_normalize_dataset(train_dataset, val_dataset, test_dataset)
    
    print(">>> trainset: ", len(train_dataset))
    print(">>> valset: ", len(val_dataset))
    print(">>> testset: ", len(test_dataset))
    print("---------- Smple 3 data point from trainset ----------")
    
    for i in random.sample(range(len(train_dataset)), 2):
        print(">>> ", train_dataset[i])
    
    def collate_fn(examples):
        # Initialize lists to store sequences and labels
        aa_seqs, labels = [], []
        structure_seqs = {
            'foldseek_seq': [] if 'foldseek_seq' in args.structure_seq else None,
            'ss8_seq': [] if 'ss8_seq' in args.structure_seq else None,
            'esm3_structure_seq': [] if 'esm3_structure_seq' in args.structure_seq else None
        }

        # Process each example
        for e in examples:
            # Process amino acid sequence
            aa_seq = e["aa_seq"]
            
            # Process structure sequences if needed
            if structure_seqs['foldseek_seq'] is not None:
                foldseek_seq = e["foldseek_seq"]
            if structure_seqs['ss8_seq'] is not None:
                ss8_seq = e["ss8_seq"]

            # Format sequences based on model type
            if 'prot_bert' in args.plm_model or "prot_t5" in args.plm_model:
                aa_seq = " ".join(list(aa_seq))
                aa_seq = re.sub(r"[UZOB]", "X", aa_seq)
                if structure_seqs['foldseek_seq'] is not None:
                    foldseek_seq = " ".join(list(foldseek_seq))
                if structure_seqs['ss8_seq'] is not None:
                    ss8_seq = " ".join(list(ss8_seq))
            elif 'ankh' in args.plm_model:
                aa_seq = list(aa_seq)
                if structure_seqs['foldseek_seq'] is not None:
                    foldseek_seq = list(foldseek_seq)
                if structure_seqs['ss8_seq'] is not None:
                    ss8_seq = list(ss8_seq)

            # Append sequences to lists
            aa_seqs.append(aa_seq)
            if structure_seqs['foldseek_seq'] is not None:
                structure_seqs['foldseek_seq'].append(foldseek_seq)
            if structure_seqs['ss8_seq'] is not None:
                structure_seqs['ss8_seq'].append(ss8_seq)
            if structure_seqs['esm3_structure_seq'] is not None:
                esm3_seq = [VQVAE_SPECIAL_TOKENS["BOS"]] + e["esm3_structure_seq"] + [VQVAE_SPECIAL_TOKENS["EOS"]]
                structure_seqs['esm3_structure_seq'].append(torch.tensor(esm3_seq))
            
            labels.append(e["label"])

        # Tokenize sequences
        if 'ankh' in args.plm_model:
            aa_inputs = tokenizer.batch_encode_plus(
                aa_seqs, add_special_tokens=True, padding=True, is_split_into_words=True,return_tensors="pt"
            )
            if structure_seqs['foldseek_seq'] is not None:
                foldseek_input_ids = tokenizer.batch_encode_plus(
                    structure_seqs['foldseek_seq'], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt"
                )["input_ids"]
            if structure_seqs['ss8_seq'] is not None:
                ss8_input_ids = tokenizer.batch_encode_plus(
                    structure_seqs['ss8_seq'], add_special_tokens=True, padding=True, is_split_into_words=True, return_tensors="pt"
                )["input_ids"]
        else:
            aa_inputs = tokenizer(aa_seqs, return_tensors="pt", padding=True, truncation=True)
            if structure_seqs['foldseek_seq'] is not None:
                foldseek_input_ids = tokenizer(
                    structure_seqs['foldseek_seq'], return_tensors="pt", padding=True, truncation=True
                )["input_ids"]
            if structure_seqs['ss8_seq'] is not None:
                ss8_input_ids = tokenizer(
                    structure_seqs['ss8_seq'], return_tensors="pt", padding=True, truncation=True
                )["input_ids"]

        # Prepare output dictionary
        data_dict = {
            "aa_input_ids": aa_inputs["input_ids"],
            "attention_mask": aa_inputs["attention_mask"],
            "label": torch.as_tensor(labels, dtype=torch.float if args.problem_type == 'regression' else torch.long)
        }

        # Add structure sequences if needed
        if structure_seqs['foldseek_seq'] is not None:
            data_dict["foldseek_input_ids"] = foldseek_input_ids
        if structure_seqs['ss8_seq'] is not None:
            data_dict["ss8_input_ids"] = ss8_input_ids
        if structure_seqs['esm3_structure_seq'] is not None:
            data_dict["esm3_structure_input_ids"] = torch.nn.utils.rnn.pad_sequence(
                structure_seqs['esm3_structure_seq'],
                batch_first=True,
                padding_value=VQVAE_SPECIAL_TOKENS["PAD"]
            )

        return data_dict
    
    # Configure loss function based on problem type and loss function choice
    loss_fn_config = {
        "single_label_classification": {
            "cross_entropy": lambda: nn.CrossEntropyLoss(),
            "focal_loss": lambda: MultiClassFocalLossWithAlpha(
                num_classes=args.num_labels,
                alpha=[len(train_dataset) / sum(1 for e in train_dataset if e["label"] == i) 
                       for i in range(args.num_labels)],
                device=device
            )
        },
        "regression": {
            "default": lambda: nn.MSELoss()
        },
        "multi_label_classification": {
            "default": lambda: nn.BCEWithLogitsLoss()
        }
    }

    if args.problem_type not in loss_fn_config:
        raise ValueError(f"Unsupported problem type: {args.problem_type}")

    loss_config = loss_fn_config[args.problem_type]
    loss_key = args.loss_function if args.problem_type == "single_label_classification" else "default"

    if loss_key not in loss_config:
        raise ValueError(f"Unsupported loss function: {loss_key}")

    loss_function = loss_config[loss_key]()
    
    if args.loss_function == "focal_loss":
        print(">>> alpha: ", loss_function.alpha.tolist())
    
    # Common DataLoader parameters
    dataloader_params = {
        'num_workers': args.num_workers,
        'collate_fn': collate_fn
    }

    if args.batch_token is not None:
        # Use BatchSampler for token-based batching
        train_sampler = BatchSampler(train_token_num, args.batch_token)
        val_sampler = BatchSampler(val_token_num, args.batch_token, shuffle=False)
        test_sampler = BatchSampler(test_token_num, args.batch_token, shuffle=False)

        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **dataloader_params)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, **dataloader_params)
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, **dataloader_params)

    else:
        # Use batch_size based loading
        batch_params = {
            'batch_size': args.batch_size,
            **dataloader_params
        }

        train_loader = DataLoader(train_dataset, shuffle=True, **batch_params)
        val_loader = DataLoader(val_dataset, shuffle=False, **batch_params) 
        test_loader = DataLoader(test_dataset, shuffle=False, **batch_params)
    
    # Initialize accelerator and optimizer
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Calculate training steps and warmup steps
    num_training_steps = len(train_dataset) * args.num_epochs // args.batch_size
    if args.warmup_steps == 0:
        args.warmup_steps = num_training_steps // 10
    num_warmup_steps = args.warmup_steps

    if args.scheduler is not None:
        # Configure learning rate scheduler
        scheduler_config = {
            'linear': lambda step: min(step / num_warmup_steps, 1.0),
            'cosine': lambda step: 0.5 * (1 + math.cos(step / num_training_steps * math.pi)),
            'step': lambda step: 1.0
        }

        if args.scheduler not in scheduler_config:
            raise ValueError(f"Unsupported scheduler type: {args.scheduler}")

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, 
            scheduler_config[args.scheduler]
        )
    
        model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader, scheduler
        )
    else:
        model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader, test_loader
        )
    
    print("---------- Start Training ----------")
    train(
        args, model, plm_model, accelerator, metrics_dict, metrics_monitor_strategy_dict, 
        train_loader, val_loader, test_loader, loss_function, optimizer, device
    )
    
    if args.wandb:
        wandb.finish()