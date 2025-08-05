import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import json
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from utils import MultiChannelSegDataset, create_model
from transforms_for_hyper_training import get_val_transforms


def calculate_dice(pred, target, smooth=1e-6):
    """Calculate Dice coefficient"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    
    return dice.item()


def calculate_iou(pred, target, smooth=1e-6):
    """Calculate IoU (Intersection over Union)"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.item()


def calculate_precision(pred, target, smooth=1e-6):
    """Calculate Precision"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    
    precision = (true_positive + smooth) / (predicted_positive + smooth)
    return precision.item()


def calculate_recall(pred, target, smooth=1e-6):
    """Calculate Recall (Sensitivity)"""
    pred = pred.view(-1)
    target = target.view(-1)
    
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    
    recall = (true_positive + smooth) / (actual_positive + smooth)
    return recall.item()


def calculate_f1_score(precision, recall):
    """Calculate F1 score from precision and recall"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def optimize_threshold_post_training(model, val_loader, device):
    """Comprehensive threshold optimization after training completes"""
    
    print("Optimizing threshold on validation set...")
    model.eval()
    
    # Comprehensive threshold search
    thresholds = np.arange(0.1, 0.95, 0.02)  # Fine-grained search
    
    with torch.no_grad():
        # Collect all predictions once
        all_predictions = []
        all_targets = []
        
        print("Collecting predictions from validation set...")
        for i, batch in enumerate(val_loader):
            if i % 10 == 0:
                print(f"  Processing batch {i+1}/{len(val_loader)}")
            
            # Handle different return formats from your dataset
            try:
                    # Format: images, targets, sample_ids
                images, targets, sample_ids = batch

            except ValueError:
                print(f"Warning: Unexpected batch format with {len(batch)} items")
                continue
     
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            
            all_predictions.append(probs.cpu())
            all_targets.append(targets.cpu())
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        print(f"Collected predictions shape: {all_predictions.shape}")
        print(f"Collected targets shape: {all_targets.shape}")
        
        # Check if we actually collected any data
        if len(all_predictions) == 0:
            print("Error: No predictions collected. Check if your validation set has masks.")
            return None, None
    
    # Evaluate all thresholds
    print("Evaluating thresholds...")
    results = {}
    best_threshold = 0.5
    best_dice = 0.0
    best_f1 = 0.0
    
    for i, threshold in enumerate(thresholds):
        if i % 5 == 0:
            print(f"  Testing threshold {threshold:.2f} ({i+1}/{len(thresholds)})")
            
        binary_preds = (all_predictions > threshold).float()
        
        dice = calculate_dice(binary_preds, all_targets)
        iou = calculate_iou(binary_preds, all_targets)
        precision = calculate_precision(binary_preds, all_targets)
        recall = calculate_recall(binary_preds, all_targets)
        f1 = calculate_f1_score(precision, recall)
        
        results[threshold] = {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        if dice > best_dice:
            best_dice = dice
            best_threshold = threshold
    
    print(f"\nThreshold optimization completed!")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best Dice score: {best_dice:.4f}")
    
    return best_threshold, results


def plot_threshold_analysis(results, best_threshold, output_dir):
    """Create plots for threshold analysis"""
    
    thresholds = list(results.keys())
    dice_scores = [results[t]['dice'] for t in thresholds]
    iou_scores = [results[t]['iou'] for t in thresholds]
    precision_scores = [results[t]['precision'] for t in thresholds]
    recall_scores = [results[t]['recall'] for t in thresholds]
    f1_scores = [results[t]['f1'] for t in thresholds]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Threshold Optimization Analysis', fontsize=16)
    
    # Plot 1: Dice and IoU
    axes[0, 0].plot(thresholds, dice_scores, 'b-', label='Dice', linewidth=2)
    axes[0, 0].plot(thresholds, iou_scores, 'r-', label='IoU', linewidth=2)
    axes[0, 0].axvline(x=best_threshold, color='g', linestyle='--', 
                       label=f'Best Threshold: {best_threshold:.3f}')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('Dice and IoU vs Threshold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Precision and Recall
    axes[0, 1].plot(thresholds, precision_scores, 'b-', label='Precision', linewidth=2)
    axes[0, 1].plot(thresholds, recall_scores, 'r-', label='Recall', linewidth=2)
    axes[0, 1].axvline(x=best_threshold, color='g', linestyle='--', 
                       label=f'Best Threshold: {best_threshold:.3f}')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision and Recall vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Score
    axes[1, 0].plot(thresholds, f1_scores, 'purple', linewidth=2)
    axes[1, 0].axvline(x=best_threshold, color='g', linestyle='--', 
                       label=f'Best Threshold: {best_threshold:.3f}')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score vs Threshold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: All metrics together
    axes[1, 1].plot(thresholds, dice_scores, 'b-', label='Dice', linewidth=2)
    axes[1, 1].plot(thresholds, iou_scores, 'r-', label='IoU', linewidth=2)
    axes[1, 1].plot(thresholds, f1_scores, 'purple', label='F1', linewidth=2)
    axes[1, 1].axvline(x=best_threshold, color='g', linestyle='--', 
                       label=f'Best Threshold: {best_threshold:.3f}')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('All Metrics vs Threshold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'threshold_optimization_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Threshold analysis plot saved to: {plot_path}")
    
    plt.show()


def load_model_and_optimize_threshold(model_path, config_path, output_dir):
    """Main function to load model and optimize threshold"""
    
    # Load configuration
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert types
    config['device'] = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    config['channels'] = list(config['channels'])
    config['n_channels'] = int(config['n_channels'])
    config['n_classes'] = int(config['n_classes'])
    config['batch_size'] = int(config['batch_size'])
    config['num_workers'] = int(config['num_workers'])
    config['seed'] = int(config['seed'])
    
    # Set seeds for reproducibility (same as training)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
    
    print(f"Using device: {config['device']}")
    print(f"Using seed: {config['seed']} (same as training)")
    
    # Load model checkpoint
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=config['device'])
    
    # Debug: Check what type of checkpoint this is
    print(f"Checkpoint type: {type(checkpoint)}")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Standard checkpoint format with metadata
            print("Found structured checkpoint with metadata")
            model_state_dict = checkpoint['model_state_dict']
            
            # Extract metadata
            if 'architecture' in checkpoint and 'encoder' in checkpoint:
                architecture = checkpoint['architecture']
                encoder = checkpoint['encoder']
            elif 'config_key' in checkpoint:
                config_key = checkpoint['config_key']
                if '_' in config_key:
                    architecture, encoder = config_key.split('_', 1)
                else:
                    architecture, encoder = 'UNet', 'resnet34'  # defaults
            else:
                architecture, encoder = 'UNet', 'resnet34'  # defaults
                
            epoch = checkpoint.get('epoch', 0)
            best_dice = checkpoint.get('best_dice', 0.0)
            best_val_loss = checkpoint.get('val_loss', checkpoint.get('best_val_loss', 0.0))
            
        elif any(key.startswith('encoder.') or key.startswith('decoder.') for key in checkpoint.keys()):
            # Direct state_dict format (what you have)
            print("Found direct state_dict format")
            model_state_dict = checkpoint
            
            # Try to infer architecture and encoder from filename or use defaults
            filename = os.path.basename(model_path).lower()
            print(f"Inferring model config from filename: {filename}")
            
            # Extract architecture from filename
            if 'unet' in filename:
                architecture = 'UNet'
            elif 'fpn' in filename:
                architecture = 'FPN'
            elif 'linknet' in filename:
                architecture = 'Linknet'
            elif 'pspnet' in filename:
                architecture = 'PSPNet'
            elif 'deeplabv3' in filename:
                architecture = 'DeepLabV3'
            else:
                architecture = 'UNet'  # Default
                
            # Extract encoder from filename
            if 'resnet34' in filename:
                encoder = 'resnet34'
            elif 'resnet50' in filename:
                encoder = 'resnet50'
            elif 'resnet18' in filename:
                encoder = 'resnet18'
            elif 'efficientnet-b0' in filename:
                encoder = 'efficientnet-b0'
            elif 'efficientnet' in filename:
                encoder = 'efficientnet-b0'
            else:
                # Try to infer from the state dict keys
                if any('encoder.' in key for key in checkpoint.keys()):
                    # Look at the encoder structure to infer type
                    encoder_keys = [k for k in checkpoint.keys() if k.startswith('encoder.')]
                    if any('layer4' in key for key in encoder_keys):
                        encoder = 'resnet34'  # or resnet50, but 34 is more common
                    else:
                        encoder = 'resnet34'  # Default
                else:
                    encoder = 'resnet34'  # Default
                    
            epoch = 0  # Unknown
            best_dice = 0.0  # Unknown
            best_val_loss = 0.0  # Unknown
            
            print(f"Inferred - Architecture: {architecture}, Encoder: {encoder}")
            
        else:
            # Unknown format
            print("Warning: Unknown checkpoint format")
            print("Available keys:", list(checkpoint.keys())[:10])
            raise ValueError("Cannot determine checkpoint format")
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(checkpoint)}")
    
    print(f"Model architecture: {architecture}")
    print(f"Model encoder: {encoder}")
    if epoch > 0:
        print(f"Model was trained for {epoch} epochs")
    if best_val_loss > 0:
        print(f"Best validation loss: {best_val_loss:.4f}")
    if best_dice > 0:
        print(f"Best Dice score during training: {best_dice:.4f}")
    
    # Create the same model architecture
    print("Creating model...")
    model = create_model(encoder, architecture, config['n_channels'], config['n_classes'])
    model = model.to(config['device'])
    
    # Load model weights
    try:
        model.load_state_dict(model_state_dict)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Trying to load with strict=False...")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
            print("Model loaded with warnings!")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            raise
    
    # Create dataset (same as training)
    print("Creating dataset...")
    
    # For threshold optimization, we need validation transforms (not augmented)
    val_transform = get_val_transforms()
    
    full_dataset = MultiChannelSegDataset(
        config['train_data_dir'], 
        config['channels'],
        transform=val_transform,  # Use validation transforms, not augmented
        manual_annotation=True
    )
    
    # Create the same train/validation split as during training
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)  # Updated to match your training code (80/20 split)
    val_size = dataset_size - train_size
    
    print(f"Dataset size: {dataset_size}")
    print(f"Train size: {train_size}, Validation size: {val_size}")
    print(f"Using 80/20 train/val split (same as training)...")
    
    # Use the same random generator seed as training
    generator = torch.Generator().manual_seed(config['seed'])
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
    
    # Create validation data loader (same parameters as training)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'],
        shuffle=False,  # No shuffling for validation
        num_workers=min(config['num_workers'], 4),
        pin_memory=True,
        persistent_workers=False,
        drop_last=False
    )
    
    print(f"Validation loader created with {len(val_loader)} batches")
    
    # Run threshold optimization
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    best_threshold, threshold_results = optimize_threshold_post_training(
        model, val_loader, config['device']
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    optimization_results = {
        'model_path': model_path,
        'model_config': {
            'architecture': architecture,
            'encoder': encoder,
            'epoch_trained': epoch,
            'best_val_loss_during_training': best_val_loss,
            'best_dice_during_training': best_dice
        },
        'dataset_info': {
            'dataset_size': dataset_size,
            'train_size': train_size,
            'val_size': val_size,
            'seed_used': config['seed']
        },
        'threshold_optimization': {
            'best_threshold': best_threshold,
            'best_dice': threshold_results[best_threshold]['dice'],
            'best_iou': threshold_results[best_threshold]['iou'],
            'best_precision': threshold_results[best_threshold]['precision'],
            'best_recall': threshold_results[best_threshold]['recall'],
            'best_f1': threshold_results[best_threshold]['f1'],
            'all_thresholds': {float(k): v for k, v in threshold_results.items()}
        }
    }
    
    # Save to JSON
    results_path = os.path.join(output_dir, 'threshold_optimization_results.json')
    with open(results_path, 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Create and save plots
    plot_threshold_analysis(threshold_results, best_threshold, output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION SUMMARY")
    print("="*60)
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Metrics at best threshold:")
    print(f"  Dice:      {threshold_results[best_threshold]['dice']:.4f}")
    print(f"  IoU:       {threshold_results[best_threshold]['iou']:.4f}")
    print(f"  Precision: {threshold_results[best_threshold]['precision']:.4f}")
    print(f"  Recall:    {threshold_results[best_threshold]['recall']:.4f}")
    print(f"  F1:        {threshold_results[best_threshold]['f1']:.4f}")
    
    # Compare with training results
    training_dice = best_dice if best_dice > 0 else 0.0
    optimized_dice = threshold_results[best_threshold]['dice']
    improvement = optimized_dice - training_dice
    
    print(f"\nComparison with training:")
    if training_dice > 0:
        print(f"  Best Dice during training: {training_dice:.4f}")
        print(f"  Best Dice with optimized threshold: {optimized_dice:.4f}")
        print(f"  Improvement: {improvement:+.4f} ({improvement/training_dice*100:+.2f}%)")
    else:
        print(f"  Training Dice not available in checkpoint")
        print(f"  Best Dice with optimized threshold: {optimized_dice:.4f}")
    
    return best_threshold, threshold_results


def find_latest_model(search_dirs=None):
    """Helper function to find the most recent model file"""
    if search_dirs is None:
        # Default search directories
        search_dirs = [
            "outputs/models",
            "models",
            "checkpoints",
            ".",  # Current directory
        ]
    
    model_files = []
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            # Look for .pth files recursively
            for root, dirs, files in os.walk(search_dir):
                for file in files:
                    if file.endswith('.pth') and 'best_model' in file:
                        file_path = os.path.join(root, file)
                        model_files.append((file_path, os.path.getmtime(file_path)))
    
    if not model_files:
        print("No model files found in search directories:")
        for d in search_dirs:
            print(f"  - {d}")
        return None
    
    # Sort by modification time (most recent first)
    model_files.sort(key=lambda x: x[1], reverse=True)
    latest_model = model_files[0][0]
    
    print(f"Found {len(model_files)} model files:")
    for i, (path, mtime) in enumerate(model_files[:5]):  # Show top 5
        import time
        time_str = time.ctime(mtime)
        marker = " <- LATEST" if i == 0 else ""
        print(f"  {i+1}. {path} ({time_str}){marker}")
    
    if len(model_files) > 5:
        print(f"  ... and {len(model_files) - 5} more")
    
    return latest_model


def main():
    """Main execution function"""
    
    # Configuration - MODIFY THESE PATHS
    # Update these paths to match your actual file locations
    
    # Option 1: If you know the exact model path, use it directly
    model_path = "/Users/yifeigu/Documents/Siobhan_Lab/roots/best_model_unet_resnet34_DAPI_FITC_TRITC.pth"
    
    # Option 2: If your model is in a subdirectory from your training output, use relative path
    # model_path = "outputs/models/UNet_resnet34/best_model_loss_0.1234_dice_0.8765_epoch_15.pth"
    
    # Option 3: Auto-find the most recent model (uncomment to use)
    # model_path = find_latest_model()
    
    config_path = "/Users/yifeigu/Documents/Siobhan_Lab/roots/olympus_segmentation/endo/config_model_selection.yaml"
    output_dir = "/Users/yifeigu/Documents/Siobhan_Lab/threshold_optimization_results"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("\nLet me help you find your model files...")
        
        # Try to auto-find models
        found_model = find_latest_model([
            "/Users/yifeigu/Documents/Siobhan_Lab/roots",
            "/Users/yifeigu/Documents/Siobhan_Lab/roots/outputs",
            "/Users/yifeigu/Documents/Siobhan_Lab/roots/models",
            "/Users/yifeigu/Documents/Siobhan_Lab/roots/olympus_segmentation/endo",
        ])
        
        if found_model:
            print(f"\nSuggested model path: {found_model}")
            print("Update the model_path variable in the script with this path and run again.")
        else:
            print("\nNo model files found. Please check:")
            print("1. Did your training complete successfully?")
            print("2. Check your training output directory for .pth files")
            print("3. Update the model_path variable with the correct path")
        return
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        print("Please update the config_path variable with the correct path to your config file")
        return
    
    # Run threshold optimization
    try:
        best_threshold, results = load_model_and_optimize_threshold(
            model_path, config_path, output_dir
        )
        print(f"\n✓ Threshold optimization completed successfully!")
        print(f"✓ Results saved to {output_dir}")
        
    except Exception as e:
        print(f"Error during threshold optimization: {str(e)}")
        raise


if __name__ == '__main__':
    main()