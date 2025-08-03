import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import json
import time
from loss_functions import DiceBCELoss
from utils import ModelTrainer, MultiChannelSegDataset, EarlyStopping, cleanup_memory, monitor_gpu_memory, create_model
import yaml


def select_model_with_cv_early_stopping(dataset, config, n_folds):
    """Improved k-fold cross-validation with early stopping integration"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=config['seed'])
    
    # Get indices
    indices = list(range(len(dataset)))
    
    # Test different architectures and encoders during CV
    architectures_to_test = config['architecture_to_test']
    encoders_to_test = config['encoder_to_test']

    cv_results = {}
    detailed_metrics = {}
    
    # Ensure output directory exists
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Create models directory for saving checkpoints
    models_dir = os.path.join(output_dir, 'cv_models')
    os.makedirs(models_dir, exist_ok=True)
    
    for arch in architectures_to_test:
        for encoder in encoders_to_test:
            config_key = f"{arch}_{encoder}"
            print(f"\n{'='*60}")
            print(f"Cross-validating: {config_key}")
            print(f"{'='*60}")
            
            fold_results = []
            detailed_metrics[config_key] = {
                'architecture': arch,
                'encoder': encoder,
                'batch_size_used': config['batch_size'],
                'folds': {}
            }
            
            # Create directory for this configuration's models
            config_models_dir = os.path.join(models_dir, config_key)
            os.makedirs(config_models_dir, exist_ok=True)
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
                fold_num = fold + 1
                print(f"\nFold {fold_num}/{n_folds} for {config_key}")
                
                # Clean memory before each fold
                cleanup_memory()
                
                # Create directory for this fold's models
                fold_models_dir = os.path.join(config_models_dir, f'fold_{fold_num}')
                os.makedirs(fold_models_dir, exist_ok=True)
                
                # Initialize fold metrics storage
                detailed_metrics[config_key]['folds'][f'fold_{fold_num}'] = {
                    'train_metrics': [],
                    'val_metrics': [],
                    'epoch_times': [],
                    'learning_rates': [],
                    'best_model_path': None,
                    'early_stopped': False,
                    'early_stop_epoch': None,
                    'early_stop_reason': None
                }
                
                model = None
                trainer = None
                train_loader = None
                val_loader = None
                
                try:
                    # Create fold datasets
                    train_subset = torch.utils.data.Subset(dataset, train_idx)
                    val_subset = torch.utils.data.Subset(dataset, val_idx)
                    
                    # Data loaders with dynamic batch size
                    train_loader = DataLoader(
                        train_subset, 
                        batch_size=config['batch_size'],
                        shuffle=True, 
                        num_workers=min(config['num_workers'], 4),
                        pin_memory=True,
                        persistent_workers=False,
                        drop_last=True
                    )
                    val_loader = DataLoader(
                        val_subset, 
                        batch_size=config['batch_size'],
                        shuffle=False, 
                        num_workers=min(config['num_workers'], 4),
                        pin_memory=True,
                        persistent_workers=False,
                        drop_last=False
                    )
                    
                    # Model initialization
                    print(f"Creating model: {arch} with encoder: {encoder}")
                    model = create_model(encoder, arch, config['n_channels'], config['n_classes'])
                    model = model.to(config['device'])
                        
                    # Training setup
                    loss_fn = DiceBCELoss()
                    optimizer = optim.AdamW(model.parameters(), lr=config['default_learning_rate'])
                    trainer = ModelTrainer(model, config['device'], loss_fn, optimizer, scheduler=None)

                    # Early stopping setup
                    early_stopping = EarlyStopping(
                        patience=config['early_stopping_patience'],
                        min_delta=config['early_stopping_min_delta'],
                        mode='max'
                    )

                    # Training loop with early stopping
                    best_dice = 0.0
                    best_epoch = 0
                    max_epochs = config['model_selection_epochs']
                    
                    for epoch in range(1, max_epochs + 1):
                        epoch_start_time = time.time()
                        
                        # # Memory cleanup if needed
                        if torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                            print(f"  Epoch {epoch} - GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
                                                        # If memory usage is too high, try cleanup
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
                        
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['train_metrics'].append(train_metrics_complete)
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['val_metrics'].append(val_metrics_complete)
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['learning_rates'].append(current_lr)
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['epoch_times'].append(epoch_time)
                        
                        # Update best model if improved
                        model_saved = False
                        if val_metrics['dice'] > best_dice:
                            best_dice = val_metrics['dice']
                            best_epoch = epoch
                            
                            # Remove old model and save new best
                            if detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['best_model_path']:
                                old_path = detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['best_model_path']
                                if os.path.exists(old_path):
                                    os.remove(old_path)
                            
                            best_model_filename = f'best_model_dice_{val_metrics["dice"]:.4f}_epoch_{epoch:02d}.pth'
                            best_model_path = os.path.join(fold_models_dir, best_model_filename)
                            
                            checkpoint = {
                                'epoch': epoch, 'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'train_loss': train_loss, 'val_loss': val_loss,
                                'train_metrics': train_metrics, 'val_metrics': val_metrics,
                                'config_key': config_key, 'fold': fold_num,
                                'architecture': arch, 'encoder': encoder,
                                'best_dice': best_dice, 'best_epoch': epoch
                            }
                            
                            torch.save(checkpoint, best_model_path)
                            detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['best_model_path'] = best_model_path
                            model_saved = True
                        
                        # Early stopping check
                        improved = early_stopping(val_metrics['dice'])
                        
                        # Clean logging
                        status = "💾" if model_saved else f"({early_stopping.counter}/{early_stopping.patience})"
                        print(f"Epoch {epoch:2d} | Train: {train_loss:.4f}/{train_metrics.get('dice', 0):.4f} | "
                              f"Val: {val_loss:.4f}/{val_metrics.get('dice', 0):.4f} | Best: {best_dice:.4f} | {status}")
                        
                        # Early stopping
                        if early_stopping.early_stop:
                            print(f"Early stopping at epoch {epoch} | Best: {best_dice:.4f} (epoch {best_epoch})")
                            detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['early_stopped'] = True
                            detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['early_stop_epoch'] = epoch
                            detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['early_stop_reason'] = f"No improvement for {early_stopping.patience} epochs"
                            break
                        
                        cleanup_memory()
                    
                    # Record completion status
                    if not early_stopping.early_stop:
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['early_stopped'] = False
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['early_stop_epoch'] = None
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['early_stop_reason'] = "Completed all epochs"
                    
                    fold_results.append({
                        'fold': fold_num,
                        'best_dice': best_dice,
                        'best_epoch': best_epoch,
                        'final_metrics': val_metrics,
                        'total_epochs': len(detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['train_metrics']),
                        'batch_size_used': config['batch_size'],
                        'model_dir': fold_models_dir,
                        'best_model_path': detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['best_model_path'],
                        'early_stopped': detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['early_stopped'],
                        'early_stop_epoch': detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['early_stop_epoch'],
                        'early_stop_reason': detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['early_stop_reason']
                    })
                    
                    print(f"Fold {fold_num}: Best Dice {best_dice:.4f} at epoch {best_epoch}"
                          f"{' (Early stopped)' if detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['early_stopped'] else ''}")
                    
                except Exception as e:
                    print(f"Fold {fold_num} failed for {config_key}: {str(e)}")
                    fold_results.append({
                        'fold': fold_num,
                        'best_dice': 0.0,
                        'failed': True,
                        'error': str(e),
                        'early_stopped': False,
                        'early_stop_epoch': None,
                        'early_stop_reason': 'Training failed'
                    })
                    
                    # Store empty metrics structure for failed folds
                    detailed_metrics[config_key]['folds'][f'fold_{fold_num}'] = {
                        'train_metrics': [],
                        'val_metrics': [],
                        'epoch_times': [],
                        'learning_rates': [],
                        'best_model_path': None,
                        'failed': True,
                        'error': str(e),
                        'early_stopped': False,
                        'early_stop_epoch': None,
                        'early_stop_reason': 'Training failed'
                    }
                
                finally:
                    # Explicit cleanup after each fold
                    if model is not None:
                        del model
                    if trainer is not None:
                        del trainer
                    if train_loader is not None:
                        del train_loader
                    if val_loader is not None:
                        del val_loader
                    del train_subset, val_subset
                    
                    # Aggressive cleanup
                    cleanup_memory()
                
                # Save metrics after each fold
                save_incremental_metrics(cv_results, detailed_metrics, config_key, fold_results, output_dir)
            
            # Calculate statistics for this architecture/encoder combo
            valid_folds = [r for r in fold_results if not r.get('failed', False)]
            if valid_folds:
                dice_scores = [r['best_dice'] for r in valid_folds]
                
                # Calculate early stopping statistics
                early_stopped_folds = [r for r in valid_folds if r.get('early_stopped', False)]
                early_stop_rate = len(early_stopped_folds) / len(valid_folds) if valid_folds else 0
                avg_epochs = np.mean([r['total_epochs'] for r in valid_folds])
                
                cv_results[config_key] = {
                    'mean_dice': np.mean(dice_scores),
                    'std_dice': np.std(dice_scores),
                    'min_dice': np.min(dice_scores),
                    'max_dice': np.max(dice_scores),
                    'n_folds': len(valid_folds),
                    'fold_results': fold_results,
                    'architecture': arch,
                    'encoder': encoder,
                    'batch_size_used': config['batch_size'],
                    'models_directory': config_models_dir,
                    # Early stopping statistics
                    'early_stop_rate': early_stop_rate,
                    'avg_epochs': avg_epochs,
                    'early_stopped_folds': len(early_stopped_folds)
                }
                
                print(f"{config_key}: {cv_results[config_key]['mean_dice']:.4f}±{cv_results[config_key]['std_dice']:.4f} "
                      f"({early_stop_rate:.0%} early stopped, avg {avg_epochs:.1f} epochs)")
                
            else:
                print(f"All folds failed for {config_key}")
                cv_results[config_key] = {
                    'mean_dice': 0.0,
                    'std_dice': 0.0,
                    'min_dice': 0.0,
                    'max_dice': 0.0,
                    'n_folds': 0,
                    'fold_results': fold_results,
                    'architecture': arch,
                    'encoder': encoder,
                    'all_failed': True,
                    'batch_size_used': config['batch_size'],
                    'models_directory': config_models_dir,
                    'early_stop_rate': 0.0,
                    'avg_epochs': 0.0,
                    'early_stopped_folds': 0
                }
            
            # Save final metrics  
            save_incremental_metrics(cv_results, detailed_metrics, config_key, fold_results, output_dir, is_final=True)
            cleanup_memory()
    
    # Find best configuration
    if cv_results:
        best_config = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_dice'])
        best_results = cv_results[best_config]
        
        print(f"\n🏆 BEST: {best_config}")
        print(f"Dice: {best_results['mean_dice']:.4f}±{best_results['std_dice']:.4f}")
        print(f"Early stopping: {best_results['early_stop_rate']:.0%}, Avg epochs: {best_results['avg_epochs']:.1f}")
        
        # Save final results
        save_final_results(cv_results, detailed_metrics, output_dir)
        
        print(f"\n✓ Results saved to {output_dir}")
        return best_results, best_config
    else:
        print("All cross-validation configurations failed!")
        return None, None


def save_incremental_metrics(cv_results, detailed_metrics, current_config, fold_results, output_dir, is_final=False):
    """Save metrics incrementally"""
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
        # Save detailed metrics
        with open(os.path.join(output_dir, 'cv_detailed_metrics.json'), 'w') as f:
            json.dump(to_python_type(detailed_metrics), f, indent=2)
        
        # Save CV results
        if cv_results:
            with open(os.path.join(output_dir, 'cv_summary_results.json'), 'w') as f:
                json.dump(to_python_type(cv_results), f, indent=2)
        
    except Exception as e:
        print(f"Warning: Failed to save metrics for {current_config}: {e}")


def save_final_results(cv_results, detailed_metrics, output_dir):
    """Save final results"""
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
    
    # Save final JSON files
    with open(os.path.join(output_dir, 'cv_detailed_metrics.json'), 'w') as f:
        json.dump(to_python_type(detailed_metrics), f, indent=2)
    
    with open(os.path.join(output_dir, 'cv_summary_results.json'), 'w') as f:
        json.dump(to_python_type(cv_results), f, indent=2)


def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config_model_selection.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert types
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    for key in ['n_channels', 'n_classes', 'batch_size', 'num_workers', 
                'model_selection_epochs', 'model_selection_folds', 'patience', 'seed']:
        config[key] = int(config[key])
    config['default_learning_rate'] = float(config['default_learning_rate'])
    config['channels'] = list(config['channels'])
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create dataset
    train_dataset = MultiChannelSegDataset(config['train_data_dir'], config['channels'])

    print("Starting Cross-Validation with Early Stopping")
    print(f"Patience: {config['early_stopping_patience']} epochs")
    
    # Run CV
    cv_results, best_cv_config = select_model_with_cv_early_stopping(
        train_dataset, config, n_folds=config['model_selection_folds']
    )

    if cv_results and best_cv_config:
        best_arch, best_encoder = best_cv_config.split("_", 1)
        config['best_architecture'] = best_arch
        config['best_encoder_name'] = best_encoder
        print(f"\nSelected: {best_arch} + {best_encoder}")
    else:
        print("Cross-validation failed")

# Updated main function
def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config_model_selection.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    config['channels'] = list(config['channels'])
    config['n_channels'] = int(config['n_channels'])
    config['n_classes'] = int(config['n_classes'])
    config['batch_size'] = int(config['batch_size'])
    config['num_workers'] = int(config['num_workers'])
    config['model_selection_epochs'] = int(config['model_selection_epochs'])
    config['model_selection_folds'] = int(config['model_selection_folds'])
    config['default_learning_rate'] = float(config['default_learning_rate'])
    config['early_stopping_patience'] = int(config['early_stopping_patience'])
    config['early_stopping_min_delta'] = float(config['early_stopping_min_delta'])
    config['seed'] = int(config['seed'])
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Monitor initial GPU memory
    print("Initial GPU memory state:")
    monitor_gpu_memory()

    # Create datasets
    train_dataset = MultiChannelSegDataset(
        config['train_data_dir'], config['channels']
    )

    # Run model selection with early stopping
    print("\n" + "="*70)
    print("STEP 1: Model Selection with Early Stopping")
    print("="*70)
    cv_results, best_cv_config = select_model_with_cv_early_stopping(
        train_dataset, config, n_folds=config['model_selection_folds']
    )

    if cv_results and best_cv_config:
        best_arch, best_encoder = best_cv_config.split("_", 1)
        config['best_architecture'] = best_arch
        config['best_encoder_name'] = best_encoder
        print(f"\nSelected: {best_arch} + {best_encoder}")
    else:
        print("Cross-validation failed")
    # Save final results
    save_final_results(cv_results, {}, config['output_dir'])
    print("\nModel selection completed!")
    print(f"Results saved to: {config['output_dir']}")
    print(f"Best Architecture: {config['best_architecture']}")
    print(f"Best Encoder: {config['best_encoder_name']}")
    print(f"CV Results: {cv_results}")

if __name__ == '__main__':
    main()

