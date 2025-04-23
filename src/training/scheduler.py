from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

def create_scheduler(args, optimizer, train_loader):
    if not args.scheduler:
        return None
        
    num_training_steps = args.num_epochs * len(train_loader)
    num_warmup_steps = args.warmup_steps or num_training_steps // 10
    
    scheduler_dict = {
        'linear': lambda: get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        ),
        'cosine': lambda: get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        ),
        'step': lambda: StepLR(optimizer, step_size=30, gamma=0.1)
    }
    
    return scheduler_dict[args.scheduler]() 