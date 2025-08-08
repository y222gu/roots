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


def select_model_with_train_val_split(dataset, config):
    """Model selection with train/validation split and early stopping"""
    
    # Calculate split sizes
    dataset_size = len(dataset)
    train_size = int(0.85 * dataset_size)
    val_size = dataset_size - train_size
    
    print(f"Dataset size: {dataset_size}")
    print(f"Train size: {train_size}, Validation size: {val_size}")
    
    # Create train/val split with seeded random generator
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True, 
        num_workers=min(config['num_workers'], 4),
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        generator=generator  # For reproducible shuffling
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False, 
        num_workers=min(config['num_workers'], 4),
        pin_memory=True,
        persistent_workers=False,
        drop_last=False
    )
    
    # Test different architectures and encoders
    architectures_to_test = config['architecture_to_test']
    encoders_to_test = config['encoder_to_test']

    results = {}
    detailed_metrics = {}
    
    # Ensure output directory exists
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    # Create models directory for saving checkpoints
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    best_overall_dice = 0.0
    best_overall_config = None
    
    for arch in architectures_to_test:
        for encoder in encoders_to_test:
            config_key = f"{arch}_{encoder}"
            print(f"\n{'='*60}")
            print(f"Training: {config_key}")
            print(f"{'='*60}")
            
            # Initialize metrics storage
            detailed_metrics[config_key] = {
                'architecture': arch,
                'encoder': encoder,
                'batch_size_used': config['batch_size'],
                'train_metrics': [],
                'val_metrics': [],
                'epoch_times': [],
                'learning_rates': [],
                'best_model_path': None,
                'early_stopped': False,
                'early_stop_epoch': None,
                'early_stop_reason': None,
                'train_size': train_size,
                'val_size': val_size
            }
            
            # Create directory for this configuration's models
            config_models_dir = os.path.join(models_dir, config_key)
            os.makedirs(config_models_dir, exist_ok=True)
            
            model = None
            trainer = None
            
            try:
                # Clean memory before each configuration
                cleanup_memory()
                
                # Model initialization
                print(f"Creating model: {arch} with encoder: {encoder}")
                model = create_model(encoder, arch, config['n_channels'], config['n_classes'])
                model = model.to(config['device'])
                    
                # Training setup
                loss_fn = DiceBCELoss()
                optimizer = optim.AdamW(
                    model.parameters(),
                    lr=config['default_learning_rate']
                )

                trainer = ModelTrainer(model, config['device'], loss_fn, optimizer, scheduler=None)

                # Early stopping setup - using 'min' mode for validation loss
                early_stopping = EarlyStopping(
                    patience=config['early_stopping_patience'],
                    min_delta=config['early_stopping_min_delta'],
                    mode='min'  # Changed to 'min' for loss (lower is better)
                )

                # Training loop with early stopping
                best_dice = 0.0
                best_val_loss = float('inf')  # Track best (lowest) validation loss
                best_epoch = 0
                max_epochs = config['model_selection_epochs']
                
                for epoch in range(1, max_epochs + 1):
                    epoch_start_time = time.time()
                    
                    # Memory monitoring
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
                    
                    detailed_metrics[config_key]['train_metrics'].append(train_metrics_complete)
                    detailed_metrics[config_key]['val_metrics'].append(val_metrics_complete)
                    detailed_metrics[config_key]['learning_rates'].append(current_lr)
                    detailed_metrics[config_key]['epoch_times'].append(epoch_time)
                    
                    # Update best model if validation loss improved (lower is better)
                    model_saved = False
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_dice = val_metrics['dice']  # Update dice for this best loss model
                        best_epoch = epoch
                        
                        # Remove old model and save new best
                        if detailed_metrics[config_key]['best_model_path']:
                            old_path = detailed_metrics[config_key]['best_model_path']
                            if os.path.exists(old_path):
                                os.remove(old_path)
                        
                        best_model_filename = f'best_model_loss_{val_loss:.4f}_dice_{val_metrics["dice"]:.4f}_epoch_{epoch:02d}.pth'
                        best_model_path = os.path.join(config_models_dir, best_model_filename)
                        
                        checkpoint = {
                            'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'train_loss': train_loss, 'val_loss': val_loss,
                            'train_metrics': train_metrics, 'val_metrics': val_metrics,
                            'config_key': config_key,
                            'architecture': arch, 'encoder': encoder,
                            'best_val_loss': best_val_loss, 'best_dice': best_dice, 'best_epoch': epoch,
                            'train_size': train_size, 'val_size': val_size
                        }
                        
                        torch.save(checkpoint, best_model_path)
                        detailed_metrics[config_key]['best_model_path'] = best_model_path
                        model_saved = True
                    
                    save_incremental_metrics(results, detailed_metrics, config_key, output_dir)

                    # Early stopping check - using validation loss
                    improved = early_stopping(val_loss)
                    
                    # Clean logging
                    status = "💾" if model_saved else f"({early_stopping.counter}/{early_stopping.patience})"
                    print(f"Epoch {epoch:2d} | Train: {train_loss:.4f}/{train_metrics.get('dice', 0):.4f} | "
                          f"Val: {val_loss:.4f}/{val_metrics.get('dice', 0):.4f} | Best Loss: {best_val_loss:.4f} | {status}")
                    
                    # Early stopping
                    if early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch} (val_loss plateau) | Best loss: {best_val_loss:.4f} (epoch {best_epoch})")
                        detailed_metrics[config_key]['early_stopped'] = True
                        detailed_metrics[config_key]['early_stop_epoch'] = epoch
                        detailed_metrics[config_key]['early_stop_reason'] = f"Val loss no improvement for {early_stopping.patience} epochs"
                        break
                    
                    cleanup_memory()
                
                # Record completion status
                if not early_stopping.early_stop:
                    detailed_metrics[config_key]['early_stopped'] = False
                    detailed_metrics[config_key]['early_stop_epoch'] = None
                    detailed_metrics[config_key]['early_stop_reason'] = "Completed all epochs"
                
                # Store results for this configuration
                results[config_key] = {
                    'architecture': arch,
                    'encoder': encoder,
                    'best_val_loss': best_val_loss,
                    'best_dice': best_dice,
                    'best_epoch': best_epoch,
                    'final_metrics': val_metrics,
                    'total_epochs': len(detailed_metrics[config_key]['train_metrics']),
                    'batch_size_used': config['batch_size'],
                    'model_dir': config_models_dir,
                    'best_model_path': detailed_metrics[config_key]['best_model_path'],
                    'early_stopped': detailed_metrics[config_key]['early_stopped'],
                    'early_stop_epoch': detailed_metrics[config_key]['early_stop_epoch'],
                    'early_stop_reason': detailed_metrics[config_key]['early_stop_reason'],
                    'train_size': train_size,
                    'val_size': val_size
                }
                
                # Track best overall configuration (by lowest validation loss)
                if best_val_loss < best_overall_dice:  # Reusing variable but now for loss
                    best_overall_dice = best_val_loss  # Now stores best loss
                    best_overall_config = config_key
                
                print(f"{config_key}: Best Loss {best_val_loss:.4f} (Dice: {best_dice:.4f}) at epoch {best_epoch}"
                      f"{' (Early stopped)' if detailed_metrics[config_key]['early_stopped'] else ''}")
                
            except Exception as e:
                print(f"Training failed for {config_key}: {str(e)}")
                results[config_key] = {
                    'architecture': arch,
                    'encoder': encoder,
                    'best_val_loss': float('inf'),
                    'best_dice': 0.0,
                    'failed': True,
                    'error': str(e),
                    'early_stopped': False,
                    'early_stop_epoch': None,
                    'early_stop_reason': 'Training failed',
                    'train_size': train_size,
                    'val_size': val_size
                }
                
                # Store empty metrics structure for failed configurations
                detailed_metrics[config_key] = {
                    'architecture': arch,
                    'encoder': encoder,
                    'batch_size_used': config['batch_size'],
                    'train_metrics': [],
                    'val_metrics': [],
                    'epoch_times': [],
                    'learning_rates': [],
                    'best_model_path': None,
                    'failed': True,
                    'error': str(e),
                    'early_stopped': False,
                    'early_stop_epoch': None,
                    'early_stop_reason': 'Training failed',
                    'train_size': train_size,
                    'val_size': val_size
                }
            
            finally:
                # Explicit cleanup after each configuration
                if model is not None:
                    del model
                if trainer is not None:
                    del trainer
                
                # Aggressive cleanup
                cleanup_memory()
            
            # Save metrics after each configuration
            save_incremental_metrics(results, detailed_metrics, config_key, output_dir)
    
    # Find and return best configuration
    if results:
        valid_results = {k: v for k, v in results.items() if not v.get('failed', False)}
        
        if valid_results:
            # Select best configuration by lowest validation loss
            best_config = min(valid_results.keys(), key=lambda k: valid_results[k]['best_val_loss'])
            best_results = valid_results[best_config]
            
            print(f"\n🏆 BEST CONFIGURATION: {best_config}")
            print(f"Best Validation Loss: {best_results['best_val_loss']:.4f}")
            print(f"Corresponding Dice: {best_results['best_dice']:.4f}")
            print(f"Best Epoch: {best_results['best_epoch']}")
            print(f"Early stopped: {best_results['early_stopped']}")
            if best_results['early_stopped']:
                print(f"Early stop epoch: {best_results['early_stop_epoch']}")
            print(f"Total epochs: {best_results['total_epochs']}")
            
            # Save final results
            save_final_results(results, detailed_metrics, output_dir)
            
            print(f"\n✓ Results saved to {output_dir}")
            return best_results, best_config
        else:
            print("All configurations failed!")
            save_final_results(results, detailed_metrics, output_dir)
            return None, None
    else:
        print("No configurations were tested!")
        return None, None


def save_incremental_metrics(results, detailed_metrics, current_config, output_dir):
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
        with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w') as f:
            json.dump(to_python_type(detailed_metrics), f, indent=2)
        
        # Save results
        if results:
            with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
                json.dump(to_python_type(results), f, indent=2)
        
    except Exception as e:
        print(f"Warning: Failed to save metrics for {current_config}: {e}")


def save_final_results(results, detailed_metrics, output_dir):
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
    with open(os.path.join(output_dir, 'detailed_metrics.json'), 'w') as f:
        json.dump(to_python_type(detailed_metrics), f, indent=2)
    
    with open(os.path.join(output_dir, 'training_results.json'), 'w') as f:
        json.dump(to_python_type(results), f, indent=2)
    
    # Create a summary report
    summary = {
        'total_configurations': len(results),
        'successful_configurations': len([r for r in results.values() if not r.get('failed', False)]),
        'failed_configurations': len([r for r in results.values() if r.get('failed', False)]),
        'configurations_with_early_stopping': len([r for r in results.values() if r.get('early_stopped', False)]),
    }
    
    # Find best configuration
    valid_results = {k: v for k, v in results.items() if not v.get('failed', False)}
    if valid_results:
        # Select best configuration by lowest validation loss
        best_config = min(valid_results.keys(), key=lambda k: valid_results[k]['best_val_loss'])
        summary['best_configuration'] = {
            'config_name': best_config,
            'architecture': valid_results[best_config]['architecture'],
            'encoder': valid_results[best_config]['encoder'],
            'best_val_loss': valid_results[best_config]['best_val_loss'],
            'best_dice': valid_results[best_config]['best_dice'],
            'best_epoch': valid_results[best_config]['best_epoch'],
            'early_stopped': valid_results[best_config]['early_stopped'],
            'total_epochs': valid_results[best_config]['total_epochs']
        }
    
    with open(os.path.join(output_dir, 'training_summary.json'), 'w') as f:
        json.dump(to_python_type(summary), f, indent=2)


def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config_model_selection.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Convert types
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    config['channels'] = list(config['channels'])
    config['n_channels'] = int(config['n_channels'])
    config['n_classes'] = int(config['n_classes'])
    config['batch_size'] = int(config['batch_size'])
    config['num_workers'] = int(config['num_workers'])
    config['model_selection_epochs'] = int(config['model_selection_epochs'])
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
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Monitor initial GPU memory
    print("Initial GPU memory state:")
    monitor_gpu_memory()

    # Create dataset
    train_dataset = MultiChannelSegDataset(
        config['train_data_dir'], config['channels'],
        transform=get_augmented_transforms(), manual_annotation=True)

    # Run model selection with train/val split
    print("\n" + "="*70)
    print("STEP 1: Model Selection with Train/Validation Split and Early Stopping")
    print("="*70)
    print(f"Using 85/15 train/validation split with seed: {config['seed']}")
    print(f"Early stopping patience: {config['early_stopping_patience']} epochs (based on validation loss)")
    
    results, best_config = select_model_with_train_val_split(train_dataset, config)

    if results and best_config:
        best_arch, best_encoder = best_config.split("_", 1)
        config['best_architecture'] = best_arch
        config['best_encoder_name'] = best_encoder
        
        print(f"\n✓ Model selection completed!")
        print(f"Results saved to: {config['output_dir']}")
        print(f"Best Architecture: {best_arch}")
        print(f"Best Encoder: {best_encoder}")
        print(f"Best Validation Loss: {results['best_val_loss']:.4f}")
        print(f"Corresponding Dice Score: {results['best_dice']:.4f}")
        print(f"Best Model Path: {results['best_model_path']}")
    else:
        print("Model selection failed - no valid configurations found!")


if __name__ == '__main__':
    main()