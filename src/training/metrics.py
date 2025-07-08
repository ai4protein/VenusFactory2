import torch
from torchmetrics.classification import Accuracy, Recall, Precision, MatthewsCorrCoef, AUROC, F1Score, MatthewsCorrCoef
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryMatthewsCorrCoef, BinaryF1Score
from torchmetrics.classification import BinaryAveragePrecision, MulticlassAveragePrecision
from torchmetrics.regression import SpearmanCorrCoef, MeanSquaredError
from torchmetrics.classification import MultilabelAveragePrecision
from torchmetrics import Metric
import torch.nn.functional as F


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

class BaseResidueMetric(Metric):
    """
    An abstract base class for handling common logic in residue-level 
    classification metrics.
    - Handles data flattening.
    - Handles the application of masks.
    - Handles data filtering based on ignore_index.
    Subclasses must implement _get_metric_instance and can override _prepare_preds.
    """
    def __init__(self, num_classes: int, average: str, ignore_index: int = -1):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.average = average
        
        # The specific torchmetrics instance is created by the subclass.
        self.metric = self._get_metric_instance()

    def _get_metric_instance(self) -> Metric:
        """Subclasses must implement this method to return the correct torchmetrics instance."""
        raise NotImplementedError("Subclasses must implement _get_metric_instance")

    def _prepare_preds(self, preds: torch.Tensor, target: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Subclasses can override this method for specific prediction processing (e.g., argmax or slicing)."""
        return preds, target
        
    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
        """Generic update logic: flatten, mask, and filter."""
        # 1. Flatten the data
        preds = preds.reshape(-1, preds.shape[-1])
        target = target.reshape(-1)

        # 2. Apply the mask
        if mask is not None:
            mask = mask.reshape(-1).bool()
            preds = preds[mask]
            target = target[mask]
        
        # 3. Filter based on ignore_index
        if self.ignore_index is not None:
            valid_idx = target != self.ignore_index
            preds = preds[valid_idx]
            target = target[valid_idx]
        
        # If there is no valid data, do not update
        if target.numel() == 0:
            return
            
        # 4. Call the subclass's specific prediction processing logic
        preds, target = self._prepare_preds(preds, target)
        
        # 5. Update the internal torchmetrics instance
        self.metric.update(preds, target)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()
    

class ResidueAUPR(BaseResidueMetric):
    """Calculates Average Precision (AUPR) for residue-level classification."""
    def _get_metric_instance(self):
        if self.num_classes == 2:
            return BinaryAveragePrecision()
        else:
            return MulticlassAveragePrecision(num_classes=self.num_classes, average=self.average, ignore_index=self.ignore_index)

    def _prepare_preds(self, preds, target):
        # For binary classification, AUPR/AUROC needs the score for the positive class (logits are fine).
        if self.num_classes == 2:
            return preds[:, 1], target
        return preds, target

class ResidueAUROC(BaseResidueMetric):
    """Calculates Area Under ROC Curve (AUROC) for residue-level classification."""
    def _get_metric_instance(self):
        if self.num_classes == 2:
            return BinaryAUROC()
        else:
            # Note: torchmetrics >= 0.7 unified the AUROC API.
            return AUROC(task='multiclass', num_classes=self.num_classes, average=self.average, ignore_index=self.ignore_index)

    def _prepare_preds(self, preds, target):
        # Same logic as AUPR.
        if self.num_classes == 2:
            return preds[:, 1], target
        return preds, target


# --- Threshold-based Metrics (F1, Accuracy) ---

class ResidueF1Score(BaseResidueMetric):
    """Calculates F1-Score for residue-level classification."""
    def _get_metric_instance(self):
        if self.num_classes == 2:
            # ignore_index is not applicable to BinaryF1Score as we have already filtered manually in the base class.
            return BinaryF1Score()
        else:
            return F1Score(task='multiclass', num_classes=self.num_classes, average=self.average, ignore_index=self.ignore_index)

    def _prepare_preds(self, preds, target):
        # F1/Accuracy requires predicted class labels.
        return torch.argmax(preds, dim=1), target


class ResidueAccuracy(BaseResidueMetric):
    """Calculates Accuracy for residue-level classification."""
    def _get_metric_instance(self):
        if self.num_classes == 2:
            return BinaryAccuracy()
        else:
            return Accuracy(task='multiclass', num_classes=self.num_classes, average=self.average, ignore_index=self.ignore_index)

    def _prepare_preds(self, preds, target):
        # Same logic as F1-Score.
        return torch.argmax(preds, dim=1), target


class ResidueMCC(BaseResidueMetric):
    """Calculates Matthews Correlation Coefficient (MCC) for residue-level classification."""
    def _get_metric_instance(self):
        if self.num_classes == 2:
            return MatthewsCorrCoef(task='binary')
        else:
            return MatthewsCorrCoef(task='multiclass', num_classes=self.num_classes)

    def _prepare_preds(self, preds, target):
        # MCC requires predicted class labels.
        return torch.argmax(preds, dim=1), target


class ResidueRecall(BaseResidueMetric):
    """Calculates Recall for residue-level classification."""
    def _get_metric_instance(self):
        if self.num_classes == 2:
            # For binary recall, torchmetrics uses 'macro' as default, which is fine.
            return Recall(task='binary', average=self.average)
        else:
            return Recall(task='multiclass', num_classes=self.num_classes, average=self.average, ignore_index=self.ignore_index)

    def _prepare_preds(self, preds, target):
        # Recall requires predicted class labels.
        return torch.argmax(preds, dim=1), target

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
        elif "residue" in args.problem_type:
            # Handle residue-level classification tasks
            metric_config = _setup_residue_metrics(metric_name, args.num_labels, device)
            
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

def _setup_residue_metrics(metric_name, num_labels, device):
    """Setup metrics for residue-level classification tasks."""
    metrics_config = {
        'aupr': {
            'metric': ResidueAUPR(num_classes=num_labels, average='micro').to(device),
        },
        'auroc': {
            'metric': ResidueAUROC(num_classes=num_labels, average='micro').to(device),
        },
        'f1': {
            'metric': ResidueF1Score(num_classes=num_labels, average='micro').to(device),
        },
        'accuracy': {
            'metric': ResidueAccuracy(num_classes=num_labels, average='micro').to(device),
        },
        'mcc': {
            'metric': ResidueMCC(num_classes=num_labels, average='micro').to(device),
        },
        'recall': {
            'metric': ResidueRecall(num_classes=num_labels, average='micro').to(device),
        }
    }
    return metrics_config.get(metric_name) 