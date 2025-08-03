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
from utils import ModelTrainer, MultiChannelSegDataset
import yaml
from utils import cleanup_memory, monitor_gpu_memory, create_model

def select_model_with_cv(dataset, config, n_folds):
    """Improved k-fold cross-validation with proper memory management, model saving, and incremental metrics"""
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
                    'best_model_path': None  # Store path to best model only
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
                        num_workers=min(config['num_workers'], 4),  # Reduce workers to save memory
                        pin_memory=True,
                        persistent_workers=False,  # Disable persistent workers to save memory
                        drop_last=True  # Drop last incomplete batch to avoid size-1 batches
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
                    
                    # Model initialization with error handling
                    print(f"Creating model: {arch} with encoder: {encoder}")
                    model = create_model(encoder, arch, config['n_channels'], config['n_classes'])
                    model = model.to(config['device'])
                        
                    
                    # Training setup - Remove scheduler for CV to save memory and simplify
                    loss_fn = DiceBCELoss()
                    optimizer = optim.AdamW(model.parameters(), lr=config['default_learning_rate'])
                    
                    # No scheduler for CV phase - keeps it simple and saves memory
                    trainer = ModelTrainer(model, config['device'], loss_fn, optimizer, scheduler=None)

                    # Training loop with early memory checks
                    best_dice = 0.0
                    best_epoch = 0
                    
                    # Limit epochs for CV phase
                    max_epochs = min(config['model_selection_epochs'], 5)  # Cap at 5 epochs for CV
                    
                    for epoch in range(1, max_epochs + 1):
                        epoch_start_time = time.time()
                        
                        # Check memory before each epoch
                        if torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                            memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                            print(f"  Epoch {epoch} - GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
                            
                            # If memory usage is too high, try cleanup
                            if memory_allocated > 40:  # If using more than 40GB
                                print("  High memory usage detected, performing cleanup...")
                                cleanup_memory()

                        
                        # Training
                        train_loss, train_metrics = trainer.train_epoch(train_loader, epoch)
                        
                        # Validation
                        val_loss, val_metrics = trainer.validate_epoch(val_loader, epoch)
                        
                        
                        epoch_time = time.time() - epoch_start_time
                        current_lr = optimizer.param_groups[0]['lr']
                        
                        # Store comprehensive metrics for this epoch
                        train_metrics_complete = {
                            'epoch': epoch,
                            'loss': train_loss,
                            'lr': current_lr,
                            'epoch_time': epoch_time,
                            **train_metrics
                        }
                        
                        val_metrics_complete = {
                            'epoch': epoch,
                            'loss': val_loss,
                            'epoch_time': epoch_time,
                            **val_metrics
                        }
                        
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['train_metrics'].append(train_metrics_complete)
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['val_metrics'].append(val_metrics_complete)
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['learning_rates'].append(current_lr)
                        detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['epoch_times'].append(epoch_time)
                        
                        # Track best performance and save best model only
                        if val_metrics['dice'] > best_dice:
                            best_dice = val_metrics['dice']
                            best_epoch = epoch
                            
                            # Save/update best model for this fold
                            best_model_filename = f'best_model_dice_{val_metrics["dice"]:.4f}_epoch_{epoch:02d}.pth'
                            best_model_path = os.path.join(fold_models_dir, best_model_filename)
                            
                            # Save model state dict with additional info
                            checkpoint = {
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'train_loss': train_loss,
                                'val_loss': val_loss,
                                'train_metrics': train_metrics,
                                'val_metrics': val_metrics,
                                'config_key': config_key,
                                'fold': fold_num,
                                'architecture': arch,
                                'encoder': encoder,
                                'batch_size': config['batch_size'],
                                'best_dice': best_dice,
                                'best_epoch': epoch
                            }
                            
                            # Remove previous best model if exists
                            if detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['best_model_path'] is not None:
                                old_path = detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['best_model_path']
                                if os.path.exists(old_path):
                                    try:
                                        os.remove(old_path)
                                        print(f"         | Removed previous best model: {os.path.basename(old_path)}")
                                    except OSError:
                                        pass  # Ignore if file can't be removed
                            
                            # Save new best model
                            torch.save(checkpoint, best_model_path)
                            detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['best_model_path'] = best_model_path
                            
                            print(f"         | New best model saved: {best_model_filename}")
                        
                        # Enhanced logging with key metrics only
                        print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Dice: {train_metrics.get('dice', 0):.4f}")
                        print(f"         | Val Loss: {val_loss:.4f} | Val Dice: {val_metrics.get('dice', 0):.4f} | Time: {epoch_time:.1f}s")
                        if val_metrics['dice'] <= best_dice and epoch > 1:
                            print(f"         | Best Dice so far: {best_dice:.4f} (Epoch {best_epoch})")
                        
                        # Early cleanup after each epoch to prevent accumulation
                        if epoch < max_epochs:  # Don't cleanup after last epoch as we'll do it in finally
                            cleanup_memory()
                    
                    fold_results.append({
                        'fold': fold_num,
                        'best_dice': best_dice,
                        'best_epoch': best_epoch,
                        'final_metrics': val_metrics,
                        'total_epochs': len(detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['train_metrics']),
                        'batch_size_used': config['batch_size'],
                        'model_dir': fold_models_dir,
                        'best_model_path': detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['best_model_path']
                    })
                    
                    print(f"Fold {fold_num} Best Dice: {best_dice:.4f} (Epoch {best_epoch})")
                    best_model_name = os.path.basename(detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['best_model_path']) if detailed_metrics[config_key]['folds'][f'fold_{fold_num}']['best_model_path'] else "None"
                    print(f"Fold {fold_num} best model: {best_model_name}")
                    
                except Exception as e:
                    print(f"Fold {fold_num} failed for {config_key}: {str(e)}")
                    fold_results.append({
                        'fold': fold_num,
                        'best_dice': 0.0,
                        'failed': True,
                        'error': str(e)
                    })
                    
                    # Store empty metrics structure for failed folds
                    detailed_metrics[config_key]['folds'][f'fold_{fold_num}'] = {
                        'train_metrics': [],
                        'val_metrics': [],
                        'epoch_times': [],
                        'learning_rates': [],
                        'best_model_path': None,
                        'failed': True,
                        'error': str(e)
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
                    
                    print(f"Fold {fold_num} cleanup completed")
                
                # SAVE METRICS AFTER EACH FOLD
                print(f"Saving metrics after fold {fold_num}...")
                save_incremental_metrics(cv_results, detailed_metrics, config_key, fold_results, output_dir)
            
            # Calculate statistics for this architecture/encoder combo
            valid_folds = [r for r in fold_results if not r.get('failed', False)]
            if valid_folds:
                dice_scores = [r['best_dice'] for r in valid_folds]
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
                    'models_directory': config_models_dir
                }
                print(f"{config_key} CV Results: {cv_results[config_key]['mean_dice']:.4f} ± {cv_results[config_key]['std_dice']:.4f}")
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
                    'models_directory': config_models_dir
                }
            
            # SAVE METRICS AFTER EACH CONFIGURATION
            print(f"Saving final metrics for configuration {config_key}...")
            save_incremental_metrics(cv_results, detailed_metrics, config_key, fold_results, output_dir, is_final=True)
            
            # Final cleanup between configurations
            cleanup_memory()
            print(f"Configuration {config_key} completed")
    
    # Find best configuration
    if cv_results:
        best_config = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_dice'])
        best_results = cv_results[best_config]
        
        print(f"\n{'='*60}")
        print(f"Best CV Configuration: {best_config}")
        print(f"Mean Dice: {best_results['mean_dice']:.4f} ± {best_results['std_dice']:.4f}")
        print(f"Range: [{best_results['min_dice']:.4f}, {best_results['max_dice']:.4f}]")
        print(f"Batch size used: {best_results['batch_size_used']}")
        print(f"Models directory: {best_results['models_directory']}")
        print(f"{'='*60}")
        
        # Save final comprehensive results
        save_final_results(cv_results, detailed_metrics, output_dir)
        
        print(f"\n✓ All results saved to {output_dir}:")
        print(f"  - cv_summary_results.json: Overall CV performance")
        print(f"  - cv_detailed_metrics.json: All epoch-level metrics")
        print(f"  - cv_metrics_train.csv & cv_metrics_val.csv: Tabular data")
        print(f"  - cv_statistics_summary.txt: Statistical summary")
        print(f"  - cv_models/: All saved model checkpoints organized by config and fold")

        create_comprehensive_cv_plots(output_dir)
        
        return best_results, best_config
    else:
        print("All cross-validation configurations failed!")
        return None, None


def save_incremental_metrics(cv_results, detailed_metrics, current_config, fold_results, output_dir, is_final=False):
    """Save metrics incrementally after each fold/configuration"""
    
    # Convert all numpy types to native Python types for JSON serialization
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
        # 1. Save/update detailed metrics
        detailed_file = os.path.join(output_dir, 'cv_detailed_metrics.json')
        with open(detailed_file, 'w') as f:
            json.dump(to_python_type(detailed_metrics), f, indent=2)
        
        # 2. Save/update CV results summary
        if cv_results:
            summary_file = os.path.join(output_dir, 'cv_summary_results.json')
            with open(summary_file, 'w') as f:
                json.dump(to_python_type(cv_results), f, indent=2)
        
        # 3. Save metrics as CSV
        save_metrics_as_csv(detailed_metrics, output_dir)
        
        # 4. Generate summary statistics if this is final for a configuration
        if is_final and cv_results:
            generate_summary_statistics(cv_results, detailed_metrics, output_dir)
        
        print(f"✓ Metrics saved incrementally for {current_config}")
        
    except Exception as e:
        print(f"Warning: Failed to save incremental metrics for {current_config}: {e}")


def save_final_results(cv_results, detailed_metrics, output_dir):
    """Save final comprehensive results"""
    
    # Convert all numpy types to native Python types for JSON serialization
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
    
    # Save detailed metrics as JSON
    with open(os.path.join(output_dir, 'cv_detailed_metrics.json'), 'w') as f:
        json.dump(to_python_type(detailed_metrics), f, indent=2)
    
    # Save CV results summary
    with open(os.path.join(output_dir, 'cv_summary_results.json'), 'w') as f:
        json.dump(to_python_type(cv_results), f, indent=2)
    
    # Save metrics as CSV for easy analysis
    save_metrics_as_csv(detailed_metrics, output_dir)
    
    # Generate summary statistics
    generate_summary_statistics(cv_results, detailed_metrics, output_dir)
    
    # Create model directory summary
    create_model_directory_summary(cv_results, output_dir)


def create_model_directory_summary(cv_results, output_dir):
    """Create a summary of saved models for easy navigation"""
    
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("CROSS-VALIDATION MODEL DIRECTORY SUMMARY")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    
    # Find best configuration
    best_config = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_dice'])
    best_results = cv_results[best_config]
    
    summary_lines.append(f"BEST CONFIGURATION: {best_config}")
    summary_lines.append(f"Mean Dice Score: {best_results['mean_dice']:.4f} ± {best_results['std_dice']:.4f}")
    summary_lines.append(f"Models Directory: {best_results.get('models_directory', 'N/A')}")
    summary_lines.append("")
    
    summary_lines.append("ALL CONFIGURATIONS:")
    summary_lines.append("-" * 40)
    
    for config_key, results in sorted(cv_results.items(), key=lambda x: x[1]['mean_dice'], reverse=True):
        summary_lines.append(f"\nConfiguration: {config_key}")
        summary_lines.append(f"  Mean Dice: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
        summary_lines.append(f"  Architecture: {results['architecture']}")
        summary_lines.append(f"  Encoder: {results['encoder']}")
        summary_lines.append(f"  Models Directory: {results.get('models_directory', 'N/A')}")
        
        if 'fold_results' in results:
            summary_lines.append(f"  Fold Results:")
            for fold_result in results['fold_results']:
                if not fold_result.get('failed', False):
                    summary_lines.append(f"    Fold {fold_result['fold']}: Dice {fold_result['best_dice']:.4f} (Epoch {fold_result['best_epoch']})")
                    if fold_result.get('best_model_path'):
                        model_name = os.path.basename(fold_result['best_model_path'])
                        summary_lines.append(f"      Best Model: {model_name}")
                else:
                    summary_lines.append(f"    Fold {fold_result['fold']}: FAILED - {fold_result.get('error', 'Unknown error')}")
    
    summary_lines.append("")
    summary_lines.append("=" * 80)
    summary_lines.append("USAGE INSTRUCTIONS:")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("To load a saved model:")
    summary_lines.append("```python")
    summary_lines.append("import torch")
    summary_lines.append("checkpoint = torch.load('path/to/model.pth')")
    summary_lines.append("model.load_state_dict(checkpoint['model_state_dict'])")
    summary_lines.append("```")
    summary_lines.append("")
    summary_lines.append("Each checkpoint contains:")
    summary_lines.append("- model_state_dict: Model weights")
    summary_lines.append("- optimizer_state_dict: Optimizer state")
    summary_lines.append("- epoch: Training epoch")
    summary_lines.append("- train_loss, val_loss: Loss values")
    summary_lines.append("- train_metrics, val_metrics: All metrics")
    summary_lines.append("- config_key, fold, architecture, encoder: Metadata")
    summary_lines.append("- best_dice, best_epoch: Best performance info")
    summary_lines.append("")
    summary_lines.append("Note: Only the best model from each fold is saved to conserve disk space.")
    summary_lines.append("")
    
    # Add best models summary
    if cv_results:
        best_config = max(cv_results.keys(), key=lambda k: cv_results[k]['mean_dice'])
        summary_lines.append("=" * 80)
        summary_lines.append("QUICK ACCESS TO BEST MODELS:")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        best_results = cv_results[best_config]
        if 'fold_results' in best_results:
            summary_lines.append(f"Best Configuration: {best_config}")
            for fold_result in best_results['fold_results']:
                if not fold_result.get('failed', False) and fold_result.get('best_model_path'):
                    summary_lines.append(f"  Fold {fold_result['fold']}: {fold_result['best_model_path']}")
    
    # Save summary
    summary_file = os.path.join(output_dir, 'model_directory_summary.txt')
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))
    
    print(f"✓ Model directory summary saved to: {summary_file}")


def save_metrics_as_csv(detailed_metrics, output_dir):
    """Save detailed metrics as CSV files for easy analysis and plotting"""
    import pandas as pd
    
    train_data = []
    val_data = []
    
    for config_key, config_data in detailed_metrics.items():
        arch = config_data['architecture']
        encoder = config_data['encoder']
        
        for fold_key, fold_data in config_data['folds'].items():
            if fold_data.get('failed', False):
                continue
                
            fold_num = int(fold_key.split('_')[1])
            
            # Process training metrics
            for train_metric in fold_data['train_metrics']:
                row = {
                    'config': config_key,
                    'architecture': arch,
                    'encoder': encoder,
                    'fold': fold_num,
                    **train_metric
                }
                train_data.append(row)
            
            # Process validation metrics
            for val_metric in fold_data['val_metrics']:
                row = {
                    'config': config_key,
                    'architecture': arch,
                    'encoder': encoder,
                    'fold': fold_num,
                    **val_metric
                }
                val_data.append(row)
    
    # Save as CSV
    if train_data:
        train_df = pd.DataFrame(train_data)
        train_df.to_csv(os.path.join(output_dir, 'cv_metrics_train.csv'), index=False)
        print(f"✓ Training metrics saved: {len(train_data)} records")
    
    if val_data:
        val_df = pd.DataFrame(val_data)
        val_df.to_csv(os.path.join(output_dir, 'cv_metrics_val.csv'), index=False)
        print(f"✓ Validation metrics saved: {len(val_data)} records")


def create_comprehensive_cv_plots(output_dir):
    """Create comprehensive plots showing all metrics for each architecture-encoder combination"""
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Set publication style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })
    
    try:
        # Load data
        train_df = pd.read_csv(os.path.join(output_dir, 'cv_metrics_train.csv'))
        val_df = pd.read_csv(os.path.join(output_dir, 'cv_metrics_val.csv'))
        
        # Get all metric columns (excluding metadata columns)
        metadata_cols = ['config', 'architecture', 'encoder', 'fold', 'epoch', 'lr', 'epoch_time']
        train_metric_cols = [col for col in train_df.columns if col not in metadata_cols]
        val_metric_cols = [col for col in val_df.columns if col not in metadata_cols and col != 'lr']
        
        # Get all unique configurations
        configs = train_df['config'].unique()
        
        print(f"Found metrics: Train={train_metric_cols}, Val={val_metric_cols}")
        print(f"Found configurations: {list(configs)}")
        
        # Create individual plots for each configuration
        for config in configs:
            print(f"Creating plots for {config}...")
            
            config_train = train_df[train_df['config'] == config]
            config_val = val_df[val_df['config'] == config]
            
            arch = config_train['architecture'].iloc[0]
            encoder = config_train['encoder'].iloc[0]
            
            # Create a comprehensive figure with subplots for all metrics
            n_metrics = len(set(train_metric_cols + val_metric_cols))
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            fig.suptitle(f'Training Progress: {arch} + {encoder}', fontsize=16, y=0.98)
            
            all_metrics = list(set(train_metric_cols + val_metric_cols))
            
            for idx, metric in enumerate(all_metrics):
                if idx >= len(axes):
                    break
                    
                ax = axes[idx]
                
                # Plot individual fold curves (lighter lines)
                folds = config_train['fold'].unique()
                colors = plt.cm.Set3(np.linspace(0, 1, len(folds)))
                
                for fold_idx, fold in enumerate(folds):
                    fold_train = config_train[config_train['fold'] == fold]
                    fold_val = config_val[config_val['fold'] == fold]
                    
                    if metric in fold_train.columns:
                        ax.plot(fold_train['epoch'], fold_train[metric], 
                               color=colors[fold_idx], alpha=0.3, linewidth=1, linestyle='-')
                    
                    if metric in fold_val.columns:
                        ax.plot(fold_val['epoch'], fold_val[metric], 
                               color=colors[fold_idx], alpha=0.3, linewidth=1, linestyle='--')
                
                # Plot mean curves (bold lines)
                if metric in config_train.columns:
                    mean_train = config_train.groupby('epoch')[metric].mean()
                    std_train = config_train.groupby('epoch')[metric].std()
                    ax.plot(mean_train.index, mean_train.values, 'b-', linewidth=2.5, label=f'Train {metric}')
                    ax.fill_between(mean_train.index, 
                                   mean_train.values - std_train.values,
                                   mean_train.values + std_train.values,
                                   alpha=0.2, color='blue')
                
                if metric in config_val.columns:
                    mean_val = config_val.groupby('epoch')[metric].mean()
                    std_val = config_val.groupby('epoch')[metric].std()
                    ax.plot(mean_val.index, mean_val.values, 'r--', linewidth=2.5, label=f'Val {metric}')
                    ax.fill_between(mean_val.index,
                                   mean_val.values - std_val.values,
                                   mean_val.values + std_val.values,
                                   alpha=0.2, color='red')
                
                ax.set_title(f'{metric.title()}', fontsize=11)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.title())
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                
                # Format y-axis based on metric type
                if metric == 'loss':
                    ax.set_yscale('log')
                elif metric in ['dice', 'iou', 'precision', 'recall', 'f1', 'accuracy']:
                    ax.set_ylim(0, 1)
            
            # Hide unused subplots
            for idx in range(len(all_metrics), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'cv_detailed_{config}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create summary comparison plots
        print("Creating summary comparison plots...")
        
        # 1. Final performance comparison (box plots for each metric)
        metrics_to_compare = ['dice', 'iou', 'loss']  # Add other metrics as needed
        available_metrics = [m for m in metrics_to_compare if m in val_df.columns]
        
        if available_metrics:
            fig, axes = plt.subplots(1, len(available_metrics), figsize=(6*len(available_metrics), 6))
            if len(available_metrics) == 1:
                axes = [axes]
            
            for idx, metric in enumerate(available_metrics):
                # Get final epoch values for each fold
                final_vals = []
                config_labels = []
                
                for config in configs:
                    config_data = val_df[val_df['config'] == config]
                    for fold in config_data['fold'].unique():
                        fold_data = config_data[config_data['fold'] == fold]
                        if len(fold_data) > 0:
                            if metric == 'loss':
                                # For loss, take minimum value
                                final_vals.append(fold_data[metric].min())
                            else:
                                # For other metrics, take maximum value
                                final_vals.append(fold_data[metric].max())
                            config_labels.append(config)
                
                comparison_df = pd.DataFrame({
                    'Configuration': config_labels,
                    metric.title(): final_vals
                })
                
                sns.boxplot(data=comparison_df, x='Configuration', y=metric.title(), ax=axes[idx])
                axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=45, ha='right')
                axes[idx].set_title(f'Final {metric.title()} Comparison')
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cv_performance_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Learning rate progression
        if 'lr' in train_df.columns:
            plt.figure(figsize=(12, 6))
            for config in configs:
                config_data = train_df[train_df['config'] == config]
                mean_lr = config_data.groupby('epoch')['lr'].mean()
                plt.plot(mean_lr.index, mean_lr.values, linewidth=2, label=config, marker='o', markersize=3)
            
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedules Comparison')
            plt.yscale('log')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cv_learning_rates.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Training time analysis
        if 'epoch_time' in train_df.columns:
            plt.figure(figsize=(10, 6))
            
            time_data = []
            for config in configs:
                config_data = train_df[train_df['config'] == config]
                mean_time = config_data['epoch_time'].mean()
                time_data.append({'Configuration': config, 'Mean Epoch Time (s)': mean_time})
            
            time_df = pd.DataFrame(time_data)
            sns.barplot(data=time_df, x='Configuration', y='Mean Epoch Time (s)')
            plt.xticks(rotation=45, ha='right')
            plt.title('Training Time Comparison')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cv_training_times.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Comprehensive plots saved to {output_dir}:")
        print(f"  - cv_detailed_[config].png: Individual configuration plots")
        print(f"  - cv_performance_comparison.png: Performance comparison")
        print(f"  - cv_learning_rates.png: Learning rate schedules")
        print(f"  - cv_training_times.png: Training time comparison")
        
    except Exception as e:
        print(f"Error creating plots: {e}")
        import traceback
        traceback.print_exc()


def generate_summary_statistics(cv_results, detailed_metrics, output_dir):
    """Generate a comprehensive summary of the cross-validation results"""
    
    summary_lines = []
    summary_lines.append("CROSS-VALIDATION RESULTS SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("")
    
    # Overall ranking
    summary_lines.append("CONFIGURATION RANKING (by mean Dice score):")
    summary_lines.append("-" * 40)
    
    ranked_configs = sorted(cv_results.items(), key=lambda x: x[1]['mean_dice'], reverse=True)
    
    for rank, (config_key, results) in enumerate(ranked_configs, 1):
        if results.get('all_failed', False):
            status = "FAILED"
        else:
            status = f"{results['mean_dice']:.4f} ± {results['std_dice']:.4f}"
        
        summary_lines.append(f"{rank:2d}. {config_key:25s} | {status}")
    
    summary_lines.append("")
    
    # Detailed statistics for each configuration
    summary_lines.append("DETAILED STATISTICS:")
    summary_lines.append("-" * 40)
    
    for config_key, results in ranked_configs:
        if results.get('all_failed', False):
            continue
            
        summary_lines.append(f"\n{config_key}:")
        summary_lines.append(f"  Architecture: {results['architecture']}")
        summary_lines.append(f"  Encoder: {results['encoder']}")
        summary_lines.append(f"  Mean Dice: {results['mean_dice']:.4f}")
        summary_lines.append(f"  Std Dice: {results['std_dice']:.4f}")
        summary_lines.append(f"  Min Dice: {results['min_dice']:.4f}")
        summary_lines.append(f"  Max Dice: {results['max_dice']:.4f}")
        summary_lines.append(f"  Valid Folds: {results['n_folds']}")
        
        # Per-fold breakdown
        summary_lines.append("  Per-fold results:")
        for fold_result in results['fold_results']:
            if not fold_result.get('failed', False):
                summary_lines.append(f"    Fold {fold_result['fold']}: {fold_result['best_dice']:.4f} (epoch {fold_result.get('best_epoch', 'N/A')})")
            else:
                summary_lines.append(f"    Fold {fold_result['fold']}: FAILED")
    
    # Add metrics summary
    summary_lines.append("\n\nMETRICS COLLECTED:")
    summary_lines.append("-" * 20)
    
    # Find all unique metrics from the detailed_metrics
    all_train_metrics = set()
    all_val_metrics = set()
    
    for config_data in detailed_metrics.values():
        for fold_data in config_data['folds'].values():
            if not fold_data.get('failed', False) and fold_data['train_metrics']:
                all_train_metrics.update(fold_data['train_metrics'][0].keys())
            if not fold_data.get('failed', False) and fold_data['val_metrics']:
                all_val_metrics.update(fold_data['val_metrics'][0].keys())
    
    # Remove metadata columns
    metadata_cols = {'epoch', 'lr', 'epoch_time', 'config', 'architecture', 'encoder', 'fold'}
    train_metrics = sorted(all_train_metrics - metadata_cols)
    val_metrics = sorted(all_val_metrics - metadata_cols)
    
    summary_lines.append(f"Training metrics: {', '.join(train_metrics)}")
    summary_lines.append(f"Validation metrics: {', '.join(val_metrics)}")
    
    # Save summary
    with open(os.path.join(output_dir, 'cv_statistics_summary.txt'), 'w') as f:
        f.write('\n'.join(summary_lines))


# Updated main function with memory monitoring
def main():
    # Load configuration from config_model_selection.yaml
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

    # Step 1: Cross-validation to find best architecture and encoder combination
    print("\n" + "="*70)
    print("STEP 1: Cross-Validation for Architecture and Encoder Selection")
    print("="*70)
    
    cv_results, best_cv_config = select_model_with_cv(
        train_dataset, 
        config, 
        n_folds=config['model_selection_folds']
    )

    if cv_results and best_cv_config:
        best_arch, best_encoder = best_cv_config.split("_", 1)
        config['best_architecture'] = best_arch
        config['best_encoder_name'] = best_encoder

        print(f"Best Architecture: {config['best_architecture']}, Best Encoder: {config['best_encoder_name']}")
        print(f"CV Score: {cv_results['mean_dice']:.4f} ± {cv_results['std_dice']:.4f}")
    else:
        print("Cross-validation failed, using defaults")
        config['best_encoder_name'] = config.get('default_encoder_name', 'resnet34')
        config['best_architecture'] = config.get('default_architecture', 'Unet')
    
    print(f"Using Encoder: {config['best_encoder_name']}, Architecture: {config['best_architecture']}")

    # Final memory check
    print("\nFinal GPU memory state:")
    monitor_gpu_memory()


if __name__ == "__main__":
    main()

    # Add some system info
    # print("System Information:")
    # print(f"Python version: {os.sys.version}")
    # print(f"PyTorch version: {torch.__version__}")
    # print(f"CUDA available: {torch.cuda.is_available()}")
    # if torch.cuda.is_available():
    #     print(f"CUDA version: {torch.version.cuda}")
    #     print(f"GPU count: {torch.cuda.device_count()}")