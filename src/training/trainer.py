import os
import torch
from tqdm import tqdm
from accelerate import Accelerator
from .scheduler import create_scheduler
from .metrics import setup_metrics
from .loss_function import MultiClassFocalLossWithAlpha
import wandb

class Trainer:
    def __init__(self, args, model, plm_model, logger):
        self.args = args
        self.model = model
        self.plm_model = plm_model
        self.logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup metrics
        self.metrics_dict, self.metrics_monitor_strategy_dict = setup_metrics(args)
        
        # Setup optimizer with different learning rates
        if self.args.training_method == 'full':
            # Use a smaller learning rate for PLM
            optimizer_grouped_parameters = [
                {
                    "params": self.model.parameters(),
                    "lr": args.learning_rate
                },
                {
                    "params": self.plm_model.parameters(),
                    "lr": args.learning_rate
                }
            ]
            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        
        # Setup accelerator
        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        
        # Setup scheduler
        self.scheduler = create_scheduler(args, self.optimizer)
        
        # Setup loss function
        self.loss_fn = self._setup_loss_function()
        
        # Prepare for distributed training
        if self.args.training_method == 'full':
            self.model, self.plm_model, self.optimizer = self.accelerator.prepare(
                self.model, self.plm_model, self.optimizer
            )
        else:
            self.model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
        if self.scheduler:
            self.scheduler = self.accelerator.prepare(self.scheduler)
            
        # Training state
        self.best_val_loss = float("inf")
        self.best_val_metric_score = -float("inf")
        self.global_steps = 0
        self.early_stop_counter = 0
        
    def _setup_loss_function(self):
        if self.args.problem_type == 'regression':
            return torch.nn.MSELoss()
        elif self.args.problem_type == 'multi_label_classification':
            return torch.nn.BCEWithLogitsLoss()
        else:
            return torch.nn.CrossEntropyLoss()
    
    def train(self, train_loader, val_loader):
        """Train the model."""
        for epoch in range(self.args.num_epochs):
            self.logger.info(f"---------- Epoch {epoch} ----------")
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.logger.info(f'Epoch {epoch} Train Loss: {train_loss:.4f}')
            
            # Validation phase
            val_loss, val_metrics = self._validate(val_loader)
            
            # Handle validation results (model saving, early stopping)
            self._handle_validation_results(epoch, val_loss, val_metrics)
            
            # Early stopping check
            if self._check_early_stopping():
                break
                
    def _train_epoch(self, train_loader):
        self.model.train()
        if self.args.training_method == 'full':
            self.plm_model.train()
            
        total_loss = 0
        epoch_iterator = tqdm(train_loader, desc="Training")
        
        for batch in epoch_iterator:
            if self.args.training_method == 'full':
                with self.accelerator.accumulate(self.model, self.plm_model):
                    loss = self._training_step(batch)
                    total_loss += loss.item() * len(batch["label"])
                    epoch_iterator.set_postfix(train_loss=loss.item())
            else:
                with self.accelerator.accumulate(self.model):
                    loss = self._training_step(batch)
                    total_loss += loss.item() * len(batch["label"])
                    epoch_iterator.set_postfix(train_loss=loss.item())
                
        return total_loss / len(train_loader.dataset)
    
    def _training_step(self, batch):
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        logits = self.model(self.plm_model, batch)
        loss = self._compute_loss(logits, batch["label"])
        
        # Backward pass
        self.accelerator.backward(loss)
        
        # Gradient clipping
        if self.args.max_grad_norm > 0:
            if self.args.training_method == 'full':
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.plm_model.parameters()),
                    self.args.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.args.max_grad_norm
                )
        
        # Optimization step
        self.optimizer.step()
        if self.scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update global steps and log
        self.global_steps += 1
        self._log_training_step(loss)
        
        return loss
    
    def _validate(self, val_loader):
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            tuple: (validation_loss, validation_metrics)
        """
        self.model.eval()
        if self.args.training_method == 'full':
            self.plm_model.eval()
            
        total_loss = 0
        total_samples = 0
        
        # Reset all metrics at the start of validation
        for metric in self.metrics_dict.values():
            metric.reset()
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                logits = self.model(self.plm_model, batch)
                loss = self._compute_loss(logits, batch["label"])
                
                # Update loss statistics
                batch_size = len(batch["label"])
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Update metrics
                self._update_metrics(logits, batch["label"])
        
        # Compute average loss
        avg_loss = total_loss / total_samples
        
        # Compute final metrics
        metrics_results = {name: metric.compute().item() 
                          for name, metric in self.metrics_dict.items()}
        
        # Log validation results
        self.logger.info(f"Validation Loss: {avg_loss:.4f}")
        for name, value in metrics_results.items():
            self.logger.info(f"Validation {name}: {value:.4f}")
        
        if self.args.wandb:
            wandb.log({
                "val/loss": avg_loss,
                **{f"val/{k}": v for k, v in metrics_results.items()}
            }, step=self.global_steps)
        
        return avg_loss, metrics_results
    
    def test(self, test_loader):
        # Load best model
        self._load_best_model()
        
        # Run evaluation
        test_loss, test_metrics = self._validate(test_loader)
        
        # Log results
        self.logger.info("Test Results:")
        self.logger.info(f"Loss: {test_loss:.4f}")
        for name, value in test_metrics.items():
            self.logger.info(f"{name}: {value:.4f}")
            
        if self.args.wandb:
            wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
            wandb.log({"test/loss": test_loss})
    
    def _compute_loss(self, logits, labels):
        if self.args.problem_type == 'regression' and self.args.num_labels == 1:
            return self.loss_fn(logits.squeeze(), labels.squeeze())
        elif self.args.problem_type == 'multi_label_classification':
            return self.loss_fn(logits, labels.float())
        else:
            return self.loss_fn(logits, labels)
    
    def _update_metrics(self, logits, labels):
        """Update metrics with current batch predictions."""
        for metric in self.metrics_dict.values():
            if self.args.problem_type == 'regression' and self.args.num_labels == 1:
                metric(logits.squeeze(), labels.squeeze())
            elif self.args.problem_type == 'multi_label_classification':
                metric(logits, labels)
            else:
                metric(torch.argmax(logits, 1), labels)
    
    def _log_training_step(self, loss):
        if self.args.wandb:
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": self.optimizer.param_groups[0]['lr']
            }, step=self.global_steps)
    
    def _save_model(self, path):
        if self.args.training_method == 'full':
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'plm_state_dict': self.plm_model.state_dict()
            }, path)
        else:
            torch.save(self.model.state_dict(), path)
    
    def _load_best_model(self):
        path = os.path.join(self.args.output_dir, self.args.output_model_name)
        if self.args.training_method == 'full':
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.plm_model.load_state_dict(checkpoint['plm_state_dict'])
        else:
            self.model.load_state_dict(torch.load(path))
    
    def _handle_validation_results(self, epoch: int, val_loss: float, val_metrics: dict):
        """
        Handle validation results, including model saving and early stopping checks.
        
        Args:
            epoch: Current epoch number
            val_loss: Validation loss
            val_metrics: Dictionary of validation metrics
        """
        # Log validation results
        self.logger.info(f'Epoch {epoch} Val Loss: {val_loss:.4f}')
        for metric_name, metric_value in val_metrics.items():
            self.logger.info(f'Epoch {epoch} Val {metric_name}: {metric_value:.4f}')
        
        if self.args.wandb:
            wandb.log({
                "val/loss": val_loss,
                **{f"val/{k}": v for k, v in val_metrics.items()}
            }, step=self.global_steps)
        
        # Check if we should save the model
        should_save = False
        monitor_value = val_loss
        
        # If monitoring a specific metric
        if self.args.monitor != 'loss' and self.args.monitor in val_metrics:
            monitor_value = val_metrics[self.args.monitor]
        
        # Get the strategy (min or max) for the monitored metric
        strategy = self.metrics_monitor_strategy_dict.get(self.args.monitor, 'min')
        
        # Check if current result is better
        if strategy == 'min':
            if monitor_value < self.best_val_metric_score:
                should_save = True
                self.best_val_metric_score = monitor_value
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
        else:  # strategy == 'max'
            if monitor_value > self.best_val_metric_score:
                should_save = True
                self.best_val_metric_score = monitor_value
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
        
        # Save model if improved
        if should_save:
            self.logger.info(f"Saving model with best {self.args.monitor}: {monitor_value:.4f}")
            save_path = os.path.join(self.args.output_dir, self.args.output_model_name)
            self._save_model(save_path)

    def _check_early_stopping(self) -> bool:
        """
        Check if training should be stopped early.
        
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.args.patience > 0 and self.early_stop_counter >= self.args.patience:
            self.logger.info(f"Early stopping triggered after {self.early_stop_counter} epochs without improvement")
            return True
        return False 