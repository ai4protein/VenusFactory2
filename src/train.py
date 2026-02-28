import os
import sys
import json
import wandb
from tools.train.cli_utils.args import parse_args
from tools.train.cli_utils.logger import setup_logging, print_model_parameters
from tools.train.data.dataloader import prepare_dataloaders
from tools.train.models.model_factory import create_models, lora_factory
from tools.train.loop.trainer import Trainer

def main():
    # Parse arguments
    args = parse_args()
    
    # Setup logging and wandb
    logger = setup_logging(args)
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            entity=args.wandb_entity,
            config=vars(args)
        )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize models and tokenizer
    if args.training_method in ['plm-lora', 'plm-qlora', 'plm-dora', 'plm-adalora', 'plm-ia3']:
        model, plm_model, tokenizer = lora_factory(args)
    else:
        model, plm_model, tokenizer = create_models(args)
    print_model_parameters(model, plm_model, logger)
    
    # Prepare data with tokenizer
    train_loader, val_loader, test_loader = prepare_dataloaders(args, tokenizer, logger)
    
    # Create trainer
    trainer = Trainer(args, model, plm_model, logger, train_loader)
    
    # Train and evaluate
    trainer.train(train_loader, val_loader)
    trainer.test(test_loader)
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()