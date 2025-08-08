import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import json
import time
from loss_functions import DiceBCELoss
from utils import ModelTrainer, MultiChannelSegDataset, EarlyStopping, cleanup_memory, monitor_gpu_memory, create_model
import yaml
from transforms_for_hyper_training import get_augmented_transforms
import argparse


def load_checkpoint_and_continue_training(checkpoint_path, config, additional_epochs):
    """Load checkpoint and continue training"""
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config['device'])
    
    # Extract configuration from checkpoint
    architecture = checkpoint['architecture']
    encoder = checkpoint['encoder']
    start_epoch = checkpoint['epoch'] + 1  # Continue from next epoch
    
    print(f"Loaded checkpoint:")
    print(f"  Architecture: {architecture}")
    print(f"  Encoder: {encoder}")
    print(f"  Last completed epoch: {checkpoint['epoch']}")
    print(f"  Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(f"  Best dice score: {checkpoint['best_dice']:.4f}")
    print(f"  Will continue from epoch: {start_epoch}")
    
    # Create the same dataset split as original training
    train_dataset = MultiChannelSegDataset(
        config['train_data_dir'], config['channels'],
        transform=get_augmented_transforms(), manual_annotation=True)
    
    # Recreate the exact same train/val split using the same seed
    dataset_size = len(train_dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    
    print(f"Dataset size: {dataset_size}")
    print(f"Train size: {train_size}, Validation size: {val_size}")
    
    # Use the same seed to get identical split
    generator = torch.Generator().manual_seed(config['seed'])
    train_split, val_split = random_split(train_dataset, [train_size, val_size], generator=generator)
    
    # Create data loaders with same settings
    train_loader = DataLoader(
        train_split, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=min(config['num_workers'], 4),
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        generator=torch.Generator().manual_seed(config['seed'])  # For reproducible shuffling
    )
    val_loader = DataLoader(
        val_split, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=min(config['num_workers'], 4),
        pin_memory=True,
        persistent_workers=False,
        drop_last=False
    )
    
    # Recreate the model
    print(f"Creating model: {architecture} with encoder: {encoder}")
    model = create_model(encoder, architecture, config['n_channels'], config['n_classes'])
    model = model.to(config['device'])
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state loaded successfully")
    
    # Recreate optimizer and load its state
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['default_learning_rate']
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Optimizer state loaded successfully")
    
    # Create trainer
    loss_fn = DiceBCELoss()
    trainer = ModelTrainer(model, config['device'], loss_fn, optimizer, scheduler=None)
    
    # Setup for continued training
    best_val_loss = checkpoint['best_val_loss']
    best_dice = checkpoint['best_dice']
    best_epoch = checkpoint['best_epoch']
    
    # Create output directory for continued training
    checkpoint_dir = os.path.dirname(checkpoint_path)
    continued_dir = os.path.join(checkpoint_dir, 'continued_training')
    os.makedirs(continued_dir, exist_ok=True)
    
    # Initialize metrics tracking for continued training
    continued_metrics = {
        'original_checkpoint': checkpoint_path,
        'original_best_val_loss': best_val_loss,
        'original_best_dice': best_dice,
        'original_best_epoch': best_epoch,
        'original_last_epoch': checkpoint['epoch'],
        'continued_from_epoch': start_epoch,
        'additional_epochs': additional_epochs,
        'train_metrics': [],
        'val_metrics': [],
        'epoch_times': [],
        'learning_rates': [],
        'best_model_path': None,
        'early_stopped': False,
        'early_stop_epoch': None,
        'early_stop_reason': None,
        'improvement_found': False
    }
    
    # Setup early stopping for continued training
    early_stopping = EarlyStopping(
        patience=config['early_stopping_patience'],
        min_delta=config['early_stopping_min_delta'],
        mode='min'
    )
    
    # Initialize early stopping with current best score
    early_stopping.best_score = best_val_loss
    
    print(f"\n{'='*60}")
    print(f"CONTINUING TRAINING")
    print(f"{'='*60}")
    print(f"Starting from epoch {start_epoch} for {additional_epochs} additional epochs")
    print(f"Current best validation loss: {best_val_loss:.4f}")
    print(f"Current best dice score: {best_dice:.4f}")
    print(f"Early stopping patience: {config['early_stopping_patience']}")
    
    # Continue training loop
    end_epoch = start_epoch + additional_epochs
    improvement_found = False
    
    for epoch in range(start_epoch, end_epoch + 1):
        epoch_start_time = time.time()
        
        # Memory monitoring
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            print(f"  Epoch {epoch} - GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
            if memory_allocated > 40:  # If using more than 40GB
                print("  High memory usage detected, performing cleanup...")
                cleanup_memory()
        
        # Training & Validation
        train_loss, train_metrics = trainer.train_epoch(train_loader, epoch)
        val_loss, val_metrics = trainer.validate_epoch(val_loader, epoch)
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store metrics
        train_metrics_complete = {
            'epoch': epoch, 'loss': train_loss, 'lr': current_lr, 
            'epoch_time': epoch_time, **train_metrics
        }
        val_metrics_complete = {
            'epoch': epoch, 'loss': val_loss, 
            'epoch_time': epoch_time, **val_metrics
        }
        
        continued_metrics['train_metrics'].append(train_metrics_complete)
        continued_metrics['val_metrics'].append(val_metrics_complete)
        continued_metrics['learning_rates'].append(current_lr)
        continued_metrics['epoch_times'].append(epoch_time)
        
        # Check for improvement and save model if better
        model_saved = False
        if val_loss < best_val_loss:
            print(f"  🎉 NEW BEST MODEL! Val loss improved: {best_val_loss:.4f} → {val_loss:.4f}")
            best_val_loss = val_loss
            best_dice = val_metrics['dice']
            best_epoch = epoch
            improvement_found = True
            continued_metrics['improvement_found'] = True
            
            # Remove old continued model if exists
            if continued_metrics['best_model_path']:
                old_path = continued_metrics['best_model_path']
                if os.path.exists(old_path):
                    os.remove(old_path)
            
            # Save new best model for continued training
            best_model_filename = f'continued_best_model_loss_{val_loss:.4f}_dice_{val_metrics["dice"]:.4f}_epoch_{epoch:02d}.pth'
            best_model_path = os.path.join(continued_dir, best_model_filename)
            
            checkpoint_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'architecture': architecture,
                'encoder': encoder,
                'best_val_loss': best_val_loss,
                'best_dice': best_dice,
                'best_epoch': best_epoch,
                'original_checkpoint': checkpoint_path,
                'continued_training': True,
                'continued_from_epoch': start_epoch,
                'train_size': train_size,
                'val_size': val_size
            }
            
            torch.save(checkpoint_save, best_model_path)
            continued_metrics['best_model_path'] = best_model_path
            model_saved = True
        
        # Save incremental results
        save_continued_training_metrics(continued_metrics, continued_dir)
        
        # Early stopping check
        improved = early_stopping(val_loss)
        
        # Clean logging
        status = "💾 NEW BEST" if model_saved else f"({early_stopping.counter}/{early_stopping.patience})"
        print(f"Epoch {epoch:2d} | Train: {train_loss:.4f}/{train_metrics.get('dice', 0):.4f} | "
              f"Val: {val_loss:.4f}/{val_metrics.get('dice', 0):.4f} | Best: {best_val_loss:.4f} | {status}")
        
        # Early stopping
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch} (val_loss plateau)")
            continued_metrics['early_stopped'] = True
            continued_metrics['early_stop_epoch'] = epoch
            continued_metrics['early_stop_reason'] = f"Val loss no improvement for {early_stopping.patience} epochs"
            break
        
        cleanup_memory()
    
    # Final status
    if not early_stopping.early_stop:
        continued_metrics['early_stopped'] = False
        continued_metrics['early_stop_epoch'] = None
        continued_metrics['early_stop_reason'] = "Completed all additional epochs"
    
    # Final results summary
    print(f"\n{'='*60}")
    print(f"CONTINUED TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Original best val loss: {continued_metrics['original_best_val_loss']:.4f}")
    print(f"Final best val loss: {best_val_loss:.4f}")
    
    if improvement_found:
        improvement = continued_metrics['original_best_val_loss'] - best_val_loss
        print(f"✅ IMPROVEMENT FOUND: {improvement:.4f} loss reduction!")
        print(f"Best dice score: {best_dice:.4f}")
        print(f"Best epoch: {best_epoch}")
        print(f"Best model saved to: {continued_metrics['best_model_path']}")
    else:
        print("❌ No improvement found during continued training")
    
    print(f"Training metrics saved to: {continued_dir}")
    
    return continued_metrics, improvement_found


def save_continued_training_metrics(metrics, output_dir):
    """Save continued training metrics"""
    def to_python_type(obj):
        if isinstance(obj, dict):
            return {k: to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_python_type(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    try:
        with open(os.path.join(output_dir, 'continued_training_metrics.json'), 'w') as f:
            json.dump(to_python_type(metrics), f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save continued training metrics: {e}")


def find_best_checkpoint(results_dir):
    """Find the best checkpoint from training results"""
    results_file = os.path.join(results_dir, 'training_results.json')
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Find the best configuration (lowest validation loss)
    valid_results = {k: v for k, v in results.items() if not v.get('failed', False)}
    
    if not valid_results:
        print("No valid training results found")
        return None
    
    best_config = min(valid_results.keys(), key=lambda k: valid_results[k]['best_val_loss'])
    best_checkpoint_path = valid_results[best_config]['best_model_path']
    
    print(f"Best configuration: {best_config}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    
    return best_checkpoint_path


def main():
    
    config_file = "config_model_selection.yaml"
    checkpoint_path = r"C:\Users\yifei\Documents\data_for_publication\results\models\Unet_resnet34\best_model_loss_0.1219_dice_0.9184_epoch_202.pth"
    results_dir = r"C:\Users\yifei\Documents\data_for_publication\results_continued_training"
    additional_epochs = 50
    
    # Load configuration
    config_path = config_file
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert types
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    config['channels'] = list(config['channels'])
    config['n_channels'] = int(config['n_channels'])
    config['n_classes'] = int(config['n_classes'])
    config['batch_size'] = int(config['batch_size'])
    config['num_workers'] = int(config['num_workers'])
    config['default_learning_rate'] = float(config['default_learning_rate'])
    config['early_stopping_patience'] = int(config['early_stopping_patience'])
    config['early_stopping_min_delta'] = float(config['early_stopping_min_delta'])
    config['seed'] = int(config['seed'])
    
    # Set seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
    
    print(f"Configuration loaded from: {config_path}")
    print(f"Device: {config['device']}")
    print(f"Additional epochs to train: {additional_epochs}")
    
    # Determine checkpoint path
    if checkpoint_path and checkpoint_path.lower() != "none":
        print(f"Using specified checkpoint: {checkpoint_path}")
    elif results_dir:
        print(f"Auto-finding best checkpoint in: {results_dir}")
        checkpoint_path = find_best_checkpoint(results_dir)
        if checkpoint_path is None:
            print("Could not find best checkpoint in results directory")
            return
    else:
        print("Error: Either CHECKPOINT_PATH or RESULTS_DIR must be specified")
        print("Edit the script configuration section at the top of main() function")
        return
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file not found: {checkpoint_path}")
        return
    
    # Monitor initial GPU memory
    print("\nInitial GPU memory state:")
    monitor_gpu_memory()
    
    # Continue training
    print(f"\nContinuing training for {additional_epochs} additional epochs...")
    continued_metrics, improvement_found = load_checkpoint_and_continue_training(
        checkpoint_path, config, additional_epochs
    )
    
    print(f"\n✓ Continued training completed!")
    if improvement_found:
        print("🎉 Model performance improved during continued training!")
    else:
        print("ℹ️  No improvement found, but training completed successfully.")


if __name__ == '__main__':
    main()