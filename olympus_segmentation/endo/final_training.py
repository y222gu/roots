import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import json
from datetime import datetime
from loss_functions import DiceBCELoss, FocalTverskyLoss
from utils import ModelTrainer, EarlyStopping, MultiChannelSegDataset
from transforms_for_hyper_training import get_augmented_transforms, get_val_transforms
import yaml
import numpy as np

def train_final_model(train_dataset, test_dataset, best_params, config):
    """Train final model with best hyperparameters"""
    # Create validation split from training data
    val_size = int(len(train_dataset) * 0.2)
    train_size = len(train_dataset) - val_size
    train_subset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Set up data loaders
    train_subset.dataset.transform = get_augmented_transforms()
    val_dataset.dataset.transform = get_val_transforms()
    test_dataset.transform = get_val_transforms()
    
    # CRITICAL FIX: Use drop_last=True and ensure minimum batch size
    train_loader = DataLoader(
        train_subset, 
        batch_size=max(2, config['batch_size']),
        shuffle=True, 
        num_workers=config['num_workers'],
        drop_last=True, 
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=max(2, config['batch_size']),
        shuffle=False, 
        num_workers=config['num_workers'],
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=max(2, config['batch_size']),
        shuffle=False, 
        num_workers=config['num_workers'],
        drop_last=True,
    )
    
    # Create model
    if config['best_architecture'] == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=config['best_encoder_name'],
            encoder_weights='imagenet',
            in_channels=config['n_channels'],
            classes=config['n_classes'],
            decoder_channels=best_params.get('decoder_channels', (256, 128, 64, 32, 16))
        )
    elif config['best_architecture'] == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(
            encoder_name=config['best_encoder_name'],
            encoder_weights='imagenet',
            in_channels=config['n_channels'],
            classes=config['n_classes']
        )
    elif config['best_architecture'] == 'FPN':
        model = smp.FPN(
            encoder_name=config['best_encoder_name'],
            encoder_weights='imagenet',
            in_channels=config['n_channels'],
            classes=config['n_classes']
        )
    else:  # Default to Unet
        model = smp.Unet(
            encoder_name=config['best_encoder_name'],
            encoder_weights='imagenet',
            in_channels=config['n_channels'],
            classes=config['n_classes'],
            decoder_channels=best_params.get('decoder_channels', (256, 128, 64, 32, 16))
        )
    
    model = model.to(config['device'])
    
    # Loss function
    if best_params.get('loss') == 'FocalTversky':
        loss_fn = FocalTverskyLoss(
            alpha=best_params.get('focal_alpha', 0.5),
            beta=1-best_params.get('focal_alpha', 0.5)
        )
    else:
        loss_fn = DiceBCELoss(
            dice_weight=best_params.get('dice_weight', 1.0),
            bce_weight=best_params.get('bce_weight', 1.0)
        )
    
    # Optimizer
    lr = best_params.get('lr', 1e-4)
    weight_decay = best_params.get('weight_decay', 1e-5)
    
    if best_params.get('optimizer') == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay,
            momentum=best_params.get('momentum', 0.9)
        )
    elif best_params.get('optimizer') == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scheduler
    if best_params.get('scheduler') == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, steps_per_epoch=len(train_loader),
            epochs=config['fine_train_epochs']
        )
    elif best_params.get('scheduler') == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['fine_train_epochs']
        )
    else:
        scheduler = None
    
    # Initialize trainer
    trainer = ModelTrainer(model, config['device'], loss_fn, optimizer, scheduler)
    
    # Training history
    history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [], 'train_iou': [], 'val_iou': [],
        'train_f1': [], 'val_f1': [], 'lr': []
    }
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['patience'], min_delta=0.0001, mode='max'
    )
    best_model_path = os.path.join(
        config['output_dir'], 
        f"best_model_{config['best_encoder_name']}_{best_params.get('architecture', 'Unet')}.pth"
    )
    
    # Training loop
    print("\n" + "="*50)
    print("Training final model with best hyperparameters")
    print("="*50)
    
    for epoch in range(1, config['fine_train_epochs'] + 1):
        # Train
        train_loss, train_metrics = trainer.train_epoch(train_loader, epoch)
        
        # Validate
        val_loss, val_metrics = trainer.validate_epoch(val_loader, epoch)
        
        # Log
        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_f1'].append(train_metrics['f1'])
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(current_lr)
        
        print(f"\nEpoch {epoch}/{config['fine_train_epochs']}")
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_metrics['dice']:.4f}, "
              f"IoU: {train_metrics['iou']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"IoU: {val_metrics['iou']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # Save best model
        if early_stopping(val_metrics['dice']):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_metrics['dice'],
                'hyperparameters': best_params
            }, best_model_path)
            print(f"Saved best model (Dice: {val_metrics['dice']:.4f})")
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Test set evaluation
    print("\n" + "="*50)
    print("Evaluating on test set")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    trainer.model = model
    test_loss, test_metrics = trainer.validate_epoch(test_loader)
    
    print(f"\nTest Set Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Dice: {test_metrics['dice']:.4f}")
    print(f"IoU: {test_metrics['iou']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    
    # Save results
    results = {
        'hyperparameters': best_params,
        'training_history': history,
        'test_metrics': test_metrics,
        'best_epoch': checkpoint['epoch'],
        'timestamp': datetime.now().isoformat()
    }
    
    with open(os.path.join(config['output_dir'], 'final_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save plots
    plot_training_history(history, config['output_dir'])
    
    return model, test_metrics


def plot_training_history(history, output_dir):
    """Create comprehensive training plots"""
    epochs = history['epoch']
    
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Metrics plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Dice
    axes[0, 0].plot(epochs, history['train_dice'], 'b-', label='Train')
    axes[0, 0].plot(epochs, history['val_dice'], 'r-', label='Val')
    axes[0, 0].set_title('Dice Score')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Dice')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # IoU
    axes[0, 1].plot(epochs, history['train_iou'], 'b-', label='Train')
    axes[0, 1].plot(epochs, history['val_iou'], 'r-', label='Val')
    axes[0, 1].set_title('IoU Score')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1
    axes[1, 0].plot(epochs, history['train_f1'], 'b-', label='Train')
    axes[1, 0].plot(epochs, history['val_f1'], 'r-', label='Val')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[1, 1].plot(epochs, history['lr'], 'g-')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
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
    config['model_selection_epochs'] = int(config['model_selection_epochs'])
    config['model_selection_folds'] = int(config['model_selection_folds'])
    config['default_learning_rate'] = float(config['default_learning_rate'])
    config['patience'] = int(config['patience'])
    config['seed'] = int(config['seed'])
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create datasets
    train_dataset = MultiChannelSegDataset(
        config['train_data_dir'], config['channels']
    )
    test_dataset = MultiChannelSegDataset(
        config['test_data_dir'], config['channels']
    )

     # Step 4: Train final model
    print("\n" + "="*70)
    print("STEP 2: Training Final Model")
    print("="*70)
    final_model, test_metrics = train_final_model(
        train_dataset, test_dataset, best_params, config
    )
    
    # Step 5: Final summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"Best Architecture: {config['best_architecture']}")
    print(f"Best Encoder: {config['best_encoder_name']}")
    print(f"Final Test Dice: {test_metrics['dice']:.4f}")
    print(f"Final Test IoU: {test_metrics['iou']:.4f}")
    print(f"Results saved to: {config['output_dir']}")
    

if __name__ == '__main__':
    main()