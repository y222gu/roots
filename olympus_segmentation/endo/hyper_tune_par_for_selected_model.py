import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import optuna
import json
import yaml
from loss_functions import DiceBCELoss, FocalTverskyLoss
from utils import ModelTrainer, EarlyStopping, MultiChannelSegDataset, create_model
from transforms_for_hyper_training import get_augmented_transforms, get_val_transforms

def objective(trial, train_dataset, val_dataset, config):
    """Optuna objective function for hyperparameter optimization"""
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-3, log=True)
    
    # Loss function selection
    loss_type = trial.suggest_categorical('loss', ['DiceBCE', 'FocalTversky'])
    if loss_type == 'DiceBCE':
        dice_weight = trial.suggest_float('dice_weight', 0.5, 2.0)
        bce_weight = trial.suggest_float('bce_weight', 0.5, 2.0)
        loss_fn = DiceBCELoss(dice_weight=dice_weight, bce_weight=bce_weight)
    else:
        alpha = trial.suggest_float('focal_alpha', 0.3, 0.7)
        beta = 1 - alpha
        loss_fn = FocalTverskyLoss(alpha=alpha, beta=beta)
    
    # Optimizer selection
    optimizer_name = trial.suggest_categorical('optimizer', ['AdamW', 'SGD', 'RMSprop'])
    
    # Create data loaders with suggested augmentations
    train_dataset.transform = get_augmented_transforms(trial)
    val_dataset.transform = get_val_transforms()  # Always use validation transforms for val
    
    # CRITICAL FIX: Use drop_last=True to avoid batch size 1 issues
    train_loader = DataLoader(
        train_dataset, 
        batch_size=max(2, config['batch_size']),  # Ensure minimum batch size of 2
        shuffle=True, 
        num_workers=4,  # Reduced workers for stability
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=max(2, config['batch_size']), 
        shuffle=False, 
        num_workers=4,
        drop_last=True,
    )
    
    # Create model
    model = create_model(
        trial, config['best_encoder_name'], config['best_architecture'],
        config['n_channels'], config['n_classes']
    ).to(config['device'])
    
    # Create optimizer
    if optimizer_name == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.8, 0.99)
        optimizer = optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum
        )
    else:  # RMSprop
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler_type = trial.suggest_categorical('scheduler', ['OneCycle', 'CosineAnnealing', 'None'])
    if scheduler_type == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader), 
            epochs=config['hyper_tune_epochs']
        )
    elif scheduler_type == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['hyper_tune_epochs']
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer = ModelTrainer(model, config['device'], loss_fn, optimizer, scheduler)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'], mode='max')
    
    # Training loop
    best_dice = 0.0
    for epoch in range(config['hyper_tune_epochs']):
        train_loss, train_metrics = trainer.train_epoch(train_loader, epoch)
        val_loss, val_metrics = trainer.validate_epoch(val_loader, epoch)
        
        # Track with Optuna
        trial.report(val_metrics['dice'], epoch)
        
        # Early stopping based on dice score
        if early_stopping(val_metrics['dice']):
            best_dice = val_metrics['dice']
        
        if early_stopping.early_stop:
            break
        
        # Pruning
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_dice

def run_hyperparameter_search(train_dataset, config, n_trials):
    """Run Optuna hyperparameter optimization"""
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # split train dataset for validation
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    study.optimize(
        lambda trial: objective(trial, train_subset, val_subset, config),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Save results
    results = {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'all_trials': [
            {
                'params': t.params,
                'value': t.value,
                'state': t.state.name
            }
            for t in study.trials
        ]
    }
    
    with open(os.path.join(config['output_dir'], 'optuna_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return study.best_params


# Main training pipeline - IMPROVED
def main():
    """Main training pipeline with improved cross-validation integration"""
    
    # # ===== ADD GPU VERIFICATION AT THE VERY BEGINNING =====
    # print("=" * 70)
    # print("GPU SETUP VERIFICATION")
    # print("=" * 70)
    
    # if not torch.cuda.is_available():
    #     print("❌ CUDA not available! You are running on CPU.")
    #     print("Please check your PyTorch CUDA installation.")
    #     return
    
    # print(f"✓ CUDA available")
    # print(f"✓ GPU count: {torch.cuda.device_count()}")
    # print(f"✓ Current GPU: {torch.cuda.get_device_name()}")
    # print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
    # print("=" * 70)
    
    # Load configuration from config.yaml
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['device'] = torch.device(config['device'])
    config['channels'] = list(config['channels'])
    config['n_channels'] = int(config['n_channels'])
    config['n_classes'] = int(config['n_classes'])
    config['batch_size'] = int(config['batch_size'])
    config['num_workers'] = int(config['num_workers'])
    config['hyper_tune_epochs'] = int(config['hyper_tune_epochs'])
    config['hyper_tune_trials'] = int(config['hyper_tune_trials'])
    config['patience'] = int(config['patience'])
    config['seed'] = int(config['seed'])
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # # ===== VERIFY DEVICE ASSIGNMENT =====
    # print(f"\n✓ Config device: {config['device']}")
    # if config['device'].type == 'cpu':
    #     print("❌ WARNING: Config is set to CPU! Something is wrong.")
    #     return
    # else:
    #     print(f"✓ Config correctly set to GPU: {config['device']}")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create datasets
    train_dataset = MultiChannelSegDataset(
        config['train_data_dir'], config['channels']
    )

    test_dataset = MultiChannelSegDataset(
        config['test_data_dir'], config['channels']
    )
    
    # Hyperparameter optimization with best architecture/encoder
    print("\n" + "="*70)
    print("STEP 3: Hyperparameter Optimization")
    print("="*70)
    best_params = run_hyperparameter_search(
        train_dataset, config, n_trials=config['hyper_tune_trials']  # Reduced for faster execution
    )

    print("\n" + "="*70)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE!")
    print("="*70)
    print(f"Best Architecture: {config['best_architecture']}")
    print(f"Best Encoder: {config['best_encoder_name']}")
    # print(f"CV Score: {cv_results['mean_dice']:.4f} ± {cv_results['std_dice']:.4f}" if cv_results else "N/A")
    print(f"Results saved to: {config['output_dir']}")


if __name__ == '__main__':
    # Run the pipeline
    main()