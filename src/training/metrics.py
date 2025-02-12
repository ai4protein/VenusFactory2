import torch
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef, BinaryF1Score
from torchmetrics.regression import SpearmanCorrCoef, MeanSquaredError
from torchmetrics.classification import MultilabelAveragePrecision


def count_f1_max(pred, target):
    """
    F1 score with the optimal threshold, Copied from TorchDrug.

    This function first enumerates all possible thresholds for deciding positive and negative
    samples, and then pick the threshold with the maximal F1 score.

    Parameters:
        pred (Tensor): predictions of shape :math:`(B, N)`
        target (Tensor): binary targets of shape :math:`(B, N)`
    """

    order = pred.argsort(descending=True, dim=1)
    target = target.gather(1, order)
    precision = target.cumsum(1) / torch.ones_like(target).cumsum(1)
    recall = target.cumsum(1) / (target.sum(1, keepdim=True) + 1e-10)
    is_start = torch.zeros_like(target).bool()
    is_start[:, 0] = 1
    is_start = torch.scatter(is_start, 1, order, is_start)

    all_order = pred.flatten().argsort(descending=True)
    order = (
        order
        + torch.arange(order.shape[0], device=order.device).unsqueeze(1)
        * order.shape[1]
    )
    order = order.flatten()
    inv_order = torch.zeros_like(order)
    inv_order[order] = torch.arange(order.shape[0], device=order.device)
    is_start = is_start.flatten()[all_order]
    all_order = inv_order[all_order]
    precision = precision.flatten()
    recall = recall.flatten()
    all_precision = precision[all_order] - torch.where(
        is_start, torch.zeros_like(precision), precision[all_order - 1]
    )
    all_precision = all_precision.cumsum(0) / is_start.cumsum(0)
    all_recall = recall[all_order] - torch.where(
        is_start, torch.zeros_like(recall), recall[all_order - 1]
    )
    all_recall = all_recall.cumsum(0) / pred.shape[0]
    all_f1 = 2 * all_precision * all_recall / (all_precision + all_recall + 1e-10)
    return all_f1.max()


class MultilabelF1Max(MultilabelAveragePrecision):

    def compute(self):
        return count_f1_max(torch.cat(self.preds), torch.cat(self.target))

def setup_metrics(args):
    """Setup metrics based on problem type and specified metrics list."""
    metrics_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for metric_name in args.metrics:
        if args.problem_type == 'regression':
            metric_config = _setup_regression_metrics(metric_name, device)
        elif args.problem_type == 'single_label_classification':
            if args.num_labels == 2:
                metric_config = _setup_binary_metrics(metric_name, device)
            else:
                metric_config = _setup_multiclass_metrics(metric_name, args.num_labels, device)            
        elif args.problem_type == 'multi_label_classification':
            metric_config = _setup_multilabel_metrics(metric_name, args.num_labels, device)
            
        if metric_config:
            metrics_dict[metric_name] = metric_config['metric']
    
    # Add loss to metrics if it's the monitor metric
    if args.monitor == 'loss':
        metrics_dict['loss'] = 'loss'
        
    return metrics_dict

def _setup_regression_metrics(metric_name, device):
    metrics_config = {
        'spearman_corr': {
            'metric': SpearmanCorrCoef().to(device),
        },
        'mse': {
            'metric': MeanSquaredError().to(device),
        }
    }
    return metrics_config.get(metric_name)

def _setup_multiclass_metrics(metric_name, num_classes, device):
    metrics_config = {
        'accuracy': {
            'metric': Accuracy(task='multiclass', num_classes=num_classes).to(device),
        },
        'recall': {
            'metric': Recall(task='multiclass', num_classes=num_classes).to(device),
        },
        'precision': {
            'metric': Precision(task='multiclass', num_classes=num_classes).to(device),
        },
        'f1': {
            'metric': F1Score(task='multiclass', num_classes=num_classes).to(device),
        },
        'mcc': {
            'metric': MatthewsCorrCoef(task='multiclass', num_classes=num_classes).to(device),
        },
        'auroc': {
            'metric': AUROC(task='multiclass', num_classes=num_classes).to(device),
        }
    }
    return metrics_config.get(metric_name)

def _setup_binary_metrics(metric_name, device):
    metrics_config = {
        'accuracy': {
            'metric': BinaryAccuracy().to(device),
        },
        'recall': {
            'metric': BinaryRecall().to(device),
        },
        'precision': {
            'metric': BinaryPrecision().to(device),
        },
        'f1': {
            'metric': BinaryF1Score().to(device),
        },
        'mcc': {
            'metric': BinaryMatthewsCorrCoef().to(device),
        },
        'auroc': {
            'metric': BinaryAUROC().to(device),
        }
    }
    return metrics_config.get(metric_name)

def _setup_multilabel_metrics(metric_name, num_labels, device):
    metrics_config = {
        'f1_max': {
            'metric': MultilabelF1Max(num_labels=num_labels).to(device),
        }
    }
    return metrics_config.get(metric_name) 