import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import optuna
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb  # Optional: for experiment tracking
from tqdm import tqdm
import json
from datetime import datetime
import cv2
from torch.utils.data import Dataset
from skimage.filters import median, gaussian
from skimage.morphology import square

class DiceBCELoss(nn.Module):
    """Combined Dice and BCE loss with configurable weights"""
    def __init__(self, dice_weight=1.0, bce_weight=1.0, dice_smooth=1e-6):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode='multilabel', smooth=dice_smooth)
        self.bce = smp.losses.SoftBCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        return self.dice_weight * self.dice(logits, targets) + \
               self.bce_weight * self.bce(logits, targets)


class FocalTverskyLoss(nn.Module):
    """Alternative loss function for handling class imbalance"""
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33):
        super().__init__()
        self.loss = smp.losses.FocalTverskyLoss(
            mode='multilabel', alpha=alpha, beta=beta, gamma=gamma
        )
    
    def forward(self, logits, targets):
        return self.loss(logits, targets)


def get_val_transforms():
    """Minimal transforms for validation - FIXED: Always resize"""
    return A.Compose([
        A.Resize(1024, 1024),  # This ensures all images are the same size
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ],
        additional_targets={
            "mask": "mask"
        })

def get_augmented_transforms(trial=None):
    """Advanced augmentation pipeline for microscopy images - FIXED"""
    # Base augmentations - ALWAYS start with resize to ensure consistent size
    base_transforms = [
        A.Resize(1024, 1024),  # CRITICAL: Always resize first
        A.RandomResizedCrop(
            size=(1024, 1024),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.1)),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5),
    ]
    
    # Conditional augmentations based on hyperparameter search
    if trial:
        p_elastic = trial.suggest_float('p_elastic', 0.0, 0.5)
        p_grid = trial.suggest_float('p_grid', 0.0, 0.5)
        p_optical = trial.suggest_float('p_optical', 0.0, 0.5)
        p_noise = trial.suggest_float('p_noise', 0.0, 0.3)
        p_blur = trial.suggest_float('p_blur', 0.0, 0.3)
    else:
        # Default values
        p_elastic = 0.3
        p_grid = 0.3
        p_optical = 0.3
        p_noise = 0.2
        p_blur = 0.2
    
    advanced_transforms = [
        # Geometric distortions (important for cross-microscope generalization)
        A.ElasticTransform(alpha=120, sigma=6, p=p_elastic),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=p_grid),
        A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=p_optical),
        
        # Intensity variations (simulate different microscope settings)
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.5
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        
        # Noise and blur (simulate different imaging conditions)
        A.GaussNoise(var_limit=(10, 50), p=p_noise),
        A.GaussianBlur(blur_limit=(3, 7), p=p_blur),
        
        # Channel-wise augmentations for fluorescence
        A.ChannelShuffle(p=0.1),  # Only occasionally
        A.ChannelDropout(channel_drop_range=(1, 1), p=0.1),
    ]
    
    return A.Compose(
        base_transforms + advanced_transforms + [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ],
        additional_targets={
            "mask": "mask"
        }
    )

class MetricTracker:
    """Track and compute multiple metrics efficiently"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        self.precision_scores = []
        self.recall_scores = []
        self.f1_scores = []
        self.accuracy_scores = []
    
    def update(self, preds, targets, threshold=0.5):
        """Update metrics with batch predictions"""
        preds_binary = (preds > threshold).float()
        
        # Compute metrics
        smooth = 1e-6
        intersection = (preds_binary * targets).sum(dim=(2, 3))
        union = preds_binary.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        
        # Dice
        dice = (2 * intersection + smooth) / (union + smooth)
        self.dice_scores.extend(dice.mean(dim=1).cpu().numpy())
        
        # IoU
        iou = (intersection + smooth) / (union - intersection + smooth)
        self.iou_scores.extend(iou.mean(dim=1).cpu().numpy())
        
        # Precision, Recall, F1
        tp = intersection
        fp = (preds_binary * (1 - targets)).sum(dim=(2, 3))
        fn = ((1 - preds_binary) * targets).sum(dim=(2, 3))
        
        precision = (tp + smooth) / (tp + fp + smooth)
        recall = (tp + smooth) / (tp + fn + smooth)
        f1 = 2 * precision * recall / (precision + recall + smooth)
        
        self.precision_scores.extend(precision.mean(dim=1).cpu().numpy())
        self.recall_scores.extend(recall.mean(dim=1).cpu().numpy())
        self.f1_scores.extend(f1.mean(dim=1).cpu().numpy())
        
        # Accuracy
        correct = (preds_binary == targets).float()
        accuracy = correct.mean(dim=(1, 2, 3))
        self.accuracy_scores.extend(accuracy.cpu().numpy())
    
    def get_metrics(self):
        """Return average metrics"""
        return {
            'dice': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores),
            'precision': np.mean(self.precision_scores),
            'recall': np.mean(self.recall_scores),
            'f1': np.mean(self.f1_scores),
            'accuracy': np.mean(self.accuracy_scores)
        }


class ModelTrainer:
    """Enhanced trainer with advanced features"""
    def __init__(self, model, device, loss_fn, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.amp.GradScaler(device='cuda')  # Mixed precision training
    
    def train_epoch(self, loader, epoch_num=None):
        """Train for one epoch with mixed precision"""
        self.model.train()
        total_loss = 0.0
        metric_tracker = MetricTracker()
        
        pbar = tqdm(loader, desc=f'Training epoch {epoch_num}')
        for batch in pbar:
            # Handle both cases: with and without masks
            if len(batch) == 3:
                imgs, masks, _ = batch
                has_masks = True
            else:
                imgs, _ = batch
                has_masks = False
                continue  # Skip samples without masks during training
            
            imgs = imgs.to(self.device)
            if has_masks:
                masks = masks.to(self.device)
                # Check if masks need permutation (H,W,C -> C,H,W)
                if masks.dim() == 4 and masks.shape[-1] == 2:  # B,H,W,C
                    masks = masks.permute(0, 3, 1, 2)  # B,C,H,W
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            with torch.amp.autocast(device_type='cuda'):
                logits = self.model(imgs)
                if has_masks:
                    loss = self.loss_fn(logits, masks)
            
            # Backward pass with gradient scaling
            if has_masks:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update metrics
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    metric_tracker.update(probs, masks)
                
                total_loss += loss.item() * imgs.size(0)
                pbar.set_postfix({'loss': loss.item()})
        
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss / len(loader.dataset) if total_loss > 0 else 0
        metrics = metric_tracker.get_metrics()
        return avg_loss, metrics
    
    def validate_epoch(self, loader, epoch_num=None):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        metric_tracker = MetricTracker()
        valid_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f'Validation epoch {epoch_num}')
            for batch in pbar:
                # Handle both cases: with and without masks
                if len(batch) == 3:
                    imgs, masks, _ = batch
                    has_masks = True
                else:
                    imgs, _ = batch
                    has_masks = False
                    continue  # Skip samples without masks during validation
                
                imgs = imgs.to(self.device)
                if has_masks:
                    masks = masks.to(self.device)
                    # Check if masks need permutation (H,W,C -> C,H,W)
                    if masks.dim() == 4 and masks.shape[-1] == 2:  # B,H,W,C
                        masks = masks.permute(0, 3, 1, 2)  # B,C,H,W
                
                with torch.amp.autocast(device_type='cuda'):
                    logits = self.model(imgs)
                    if has_masks:
                        loss = self.loss_fn(logits, masks)
                
                if has_masks:
                    probs = torch.sigmoid(logits)
                    metric_tracker.update(probs, masks)
                    
                    total_loss += loss.item() * imgs.size(0)
                    valid_samples += imgs.size(0)
                    pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / valid_samples if valid_samples > 0 else 0
        metrics = metric_tracker.get_metrics()
        return avg_loss, metrics


class EarlyStopping:
    """Enhanced early stopping with relative improvement tracking"""
    def __init__(self, patience=10, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return True
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


def create_model(trial, encoder_name, n_channels, n_classes):
    """Create model with trial-suggested architecture"""
    architecture = trial.suggest_categorical(
        'architecture', ['Unet', 'UnetPlusPlus', 'DeepLabV3Plus', 'FPN']
    )
    
    decoder_channels = trial.suggest_categorical(
        'decoder_channels', 
        [(256, 128, 64, 32, 16), (512, 256, 128, 64, 32)]
    )
    
    if architecture == 'Unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=n_channels,
            classes=n_classes,
            decoder_channels=decoder_channels
        )
    elif architecture == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=n_channels,
            classes=n_classes,
            decoder_channels=decoder_channels
        )
    elif architecture == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=n_channels,
            classes=n_classes
        )
    else:  # FPN
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=n_channels,
            classes=n_classes,
            decoder_pyramid_channels=decoder_channels[0]
        )
    
    return model


def objective(trial, train_dataset, val_dataset, config):
    """Optuna objective function for hyperparameter optimization"""
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32])  # Reduced batch sizes
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
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
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2  # Reduced workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2  # Reduced workers
    )
    
    # Create model
    model = create_model(
        trial, config['encoder_name'], 
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
            epochs=config['epochs']
        )
    elif scheduler_type == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs']
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer = ModelTrainer(model, config['device'], loss_fn, optimizer, scheduler)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'], mode='max')
    
    # Training loop
    best_dice = 0.0
    for epoch in range(config['epochs']):
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


def run_hyperparameter_search(train_dataset, val_dataset, config, n_trials=50):
    """Run Optuna hyperparameter optimization"""
    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, config),
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


def train_final_model(train_dataset, val_dataset, test_dataset, best_params, config):
    """Train final model with best hyperparameters"""
    # Set up data loaders
    train_dataset.transform = get_augmented_transforms()
    val_dataset.transform = get_val_transforms()
    test_dataset.transform = get_val_transforms()
    
    batch_size = best_params.get('batch_size', 16)  # Default to smaller batch size
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=2
    )
    
    # Create model
    if best_params.get('architecture') == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=config['encoder_name'],
            encoder_weights='imagenet',
            in_channels=config['n_channels'],
            classes=config['n_classes'],
            decoder_channels=best_params.get('decoder_channels', (256, 128, 64, 32, 16))
        )
    elif best_params.get('architecture') == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(
            encoder_name=config['encoder_name'],
            encoder_weights='imagenet',
            in_channels=config['n_channels'],
            classes=config['n_classes']
        )
    elif best_params.get('architecture') == 'FPN':
        model = smp.FPN(
            encoder_name=config['encoder_name'],
            encoder_weights='imagenet',
            in_channels=config['n_channels'],
            classes=config['n_classes']
        )
    else:  # Default to Unet
        model = smp.Unet(
            encoder_name=config['encoder_name'],
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
            epochs=config['final_epochs']
        )
    elif best_params.get('scheduler') == 'CosineAnnealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['final_epochs']
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
        f"best_model_{config['encoder_name']}_{best_params.get('architecture', 'Unet')}.pth"
    )
    
    # Training loop
    print("\n" + "="*50)
    print("Training final model with best hyperparameters")
    print("="*50)
    
    for epoch in range(1, config['final_epochs'] + 1):
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
        
        print(f"\nEpoch {epoch}/{config['final_epochs']}")
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


def compare_encoders(train_dataset, val_dataset, test_dataset, config):
    """Compare different encoder architectures"""
    encoders = [
        'resnet34', 'resnet50', 'resnet101',
        'efficientnet-b0', 'efficientnet-b3', 'efficientnet-b5',
        'densenet121', 'densenet169',
        'vgg16', 'vgg19',
        'mobilenet_v2',
        'timm-efficientnet-b4'
    ]
    
    results = {}
    
    for encoder in encoders:
        print(f"\n{'='*50}")
        print(f"Testing encoder: {encoder}")
        print(f"{'='*50}")
        
        try:
            # Set transforms consistently
            train_dataset.transform = get_augmented_transforms()
            val_dataset.transform = get_val_transforms()
            test_dataset.transform = get_val_transforms()
            
            # Create model
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=config['n_channels'],
                classes=config['n_classes']
            ).to(config['device'])
            
            # Simple training setup
            loss_fn = DiceBCELoss()
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['encoder_test_epochs']
            )
            
            # Data loaders with smaller batch sizes and fewer workers
            train_loader = DataLoader(
                train_dataset, batch_size=16, shuffle=True, num_workers=2
            )
            val_loader = DataLoader(
                val_dataset, batch_size=16, shuffle=False, num_workers=2
            )
            test_loader = DataLoader(
                test_dataset, batch_size=16, shuffle=False, num_workers=2
            )
            
            # Trainer
            trainer = ModelTrainer(model, config['device'], loss_fn, optimizer, scheduler)
            
            # Quick training
            best_val_dice = 0.0
            for epoch in range(1, config['encoder_test_epochs'] + 1):
                _, train_metrics = trainer.train_epoch(train_loader, epoch)
                _, val_metrics = trainer.validate_epoch(val_loader, epoch)
                
                if val_metrics['dice'] > best_val_dice:
                    best_val_dice = val_metrics['dice']
            
            # Test evaluation
            _, test_metrics = trainer.validate_epoch(test_loader)
            
            results[encoder] = {
                'best_val_dice': best_val_dice,
                'test_metrics': test_metrics,
                'param_count': sum(p.numel() for p in model.parameters()) / 1e6  # in millions
            }
            
            print(f"Val Dice: {best_val_dice:.4f}, Test Dice: {test_metrics['dice']:.4f}")
            
        except Exception as e:
            print(f"Failed to test {encoder}: {str(e)}")
            continue
    
    # Save comparison results
    comparison_df = pd.DataFrame([
        {
            'encoder': enc,
            'val_dice': res['best_val_dice'],
            'test_dice': res['test_metrics']['dice'],
            'test_iou': res['test_metrics']['iou'],
            'params_M': res['param_count']
        }
        for enc, res in results.items()
    ])
    
    comparison_df = comparison_df.sort_values('test_dice', ascending=False)
    comparison_df.to_csv(os.path.join(config['output_dir'], 'encoder_comparison.csv'), index=False)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    x = range(len(comparison_df))
    plt.bar(x, comparison_df['test_dice'], alpha=0.7, label='Test Dice')
    plt.bar(x, comparison_df['test_iou'], alpha=0.7, label='Test IoU')
    plt.xticks(x, comparison_df['encoder'], rotation=45, ha='right')
    plt.ylabel('Score')
    plt.title('Encoder Architecture Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config['output_dir'], 'encoder_comparison.png'), dpi=300)
    plt.close()
    
    return comparison_df


def cross_validate_model(dataset, config, n_folds=5):
    """Perform k-fold cross-validation - FIXED"""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    # Get indices
    indices = list(range(len(dataset)))
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{n_folds}")
        print(f"{'='*50}")
        
        # Create fold datasets
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        
        # Create separate dataset instances for different transforms
        # This is a workaround since we can't easily change transforms on subsets
        
        # For training - need to create DataLoader with augmented transforms
        dataset.transform = get_augmented_transforms()
        train_loader = DataLoader(
            train_subset, batch_size=16, shuffle=True, num_workers=2
        )
        
        # For validation - need validation transforms
        dataset.transform = get_val_transforms()
        val_loader = DataLoader(
            val_subset, batch_size=16, shuffle=False, num_workers=2
        )
        
        # Model
        model = smp.Unet(
            encoder_name=config['encoder_name'],
            encoder_weights='imagenet',
            in_channels=config['n_channels'],
            classes=config['n_classes']
        ).to(config['device'])
        
        # Training setup
        loss_fn = DiceBCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        trainer = ModelTrainer(model, config['device'], loss_fn, optimizer)
        
        # Train
        best_dice = 0.0
        for epoch in range(1, config['cv_epochs'] + 1):
            # For training, switch to augmented transforms
            dataset.transform = get_augmented_transforms()
            _, train_metrics = trainer.train_epoch(train_loader, epoch)
            
            # For validation, switch to validation transforms
            dataset.transform = get_val_transforms()
            _, val_metrics = trainer.validate_epoch(val_loader, epoch)
            
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
        
        fold_results.append({
            'fold': fold + 1,
            'best_dice': best_dice,
            'final_metrics': val_metrics
        })
        
        print(f"Fold {fold + 1} Best Dice: {best_dice:.4f}")
    
    # Summary statistics
    dice_scores = [r['best_dice'] for r in fold_results]
    cv_results = {
        'mean_dice': np.mean(dice_scores),
        'std_dice': np.std(dice_scores),
        'fold_results': fold_results
    }
    
    print(f"\n{'='*50}")
    print(f"Cross-Validation Results")
    print(f"{'='*50}")
    print(f"Mean Dice: {cv_results['mean_dice']:.4f} ± {cv_results['std_dice']:.4f}")
    
    # Save results
    with open(os.path.join(config['output_dir'], 'cv_results.json'), 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    return cv_results


def test_generalization(model, test_datasets, config):
    """Test model generalization on multiple test sets"""
    results = {}
    
    for name, test_dataset in test_datasets.items():
        print(f"\nTesting on {name} dataset...")
        
        test_dataset.transform = get_val_transforms()
        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=2
        )
        
        # Create dummy trainer for evaluation
        trainer = ModelTrainer(
            model, config['device'], 
            DiceBCELoss(), None  # Loss and optimizer not used for eval
        )
        
        _, metrics = trainer.validate_epoch(test_loader)
        results[name] = metrics
        
        print(f"{name} Results:")
        print(f"  Dice: {metrics['dice']:.4f}")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  F1: {metrics['f1']:.4f}")
    
    return results


# --------------------- Dataset Class - FIXED ---------------------
class MultiChannelSegDataset(Dataset):
    def __init__(self, data_dir, channels, transform=None, manual_annotation=True):
        """
        data_dir: root directory (e.g. "Training/")
        channels: list of substrings to identify each channel file (['DAPI','FITC','TRITC'])
        """
        self.transform = transform
        self.channels = channels
        self.manual_annotation = manual_annotation

        # find all sample folders (those that contain at least one .ome.tif)
        self.samples = []
        for root, dirs, files in os.walk(data_dir):
            tif_files = [f for f in files if f.endswith(('.tif', '.tiff', '.ome.tif'))]
            if not tif_files:
                continue

            sample_id = os.path.basename(root)
            parent    = os.path.dirname(root)

            # find the annotation .ome.txt in the parent folder
            ann_file = next(
                (f for f in os.listdir(parent)
                 if f.startswith(sample_id) and f.endswith('.txt')),
                None
            )
            if manual_annotation and ann_file is None:
                print(f"[Skipping] no annotation for {sample_id}")
                continue

            ann_path = os.path.join(parent, ann_file) if ann_file else None
            self.samples.append((root, ann_path, sample_id))

        print(f"[Dataset] Found {len(self.samples)} samples under {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, ann_path, sid = self.samples[idx]
        image = self._load_image_stack(img_dir)         # H×W×C float32
        
        # ALWAYS preprocess the image to ensure consistent handling
        # image = self.preprocess(image)
        
        mask = self._yolo_to_inner_outer_mask(ann_path, image.shape[:2]) if ann_path else None

        if mask is not None:
            # Apply transforms if provided
            if self.transform:
                aug = self.transform(image=image, mask=mask)
                image, mask = aug['image'], aug['mask']
            else:
                # If no transforms, still need to convert to tensor and normalize
                image = torch.from_numpy(image).permute(2, 0, 1).float()  # H,W,C -> C,H,W
                mask = torch.from_numpy(mask).permute(2, 0, 1).float()    # H,W,C -> C,H,W

            # Ensure tensors are contiguous
            if isinstance(image, torch.Tensor):
                image = image.contiguous()
            if isinstance(mask, torch.Tensor):
                mask = mask.contiguous()

            return image, mask, sid
        else:
            # Handle case with no mask
            if self.transform:
                # Create a dummy mask for augmentation (will be ignored)
                dummy_mask = np.zeros((*image.shape[:2], 2), dtype=np.float32)
                aug = self.transform(image=image, mask=dummy_mask)
                image = aug['image']
            else:
                # Convert to tensor manually
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            
            if isinstance(image, torch.Tensor):
                image = image.contiguous()

            return image, sid

    def _load_image_stack(self, folder):
        imgs = []
        for ch in self.channels:
            fn = next(
                (f for f in os.listdir(folder)
                 if ch in f and f.lower().endswith(('.ome.tif', '.tif', '.tiff'))),
                None
            )
            if fn is None:
                raise FileNotFoundError(f"Channel {ch} missing in {folder}")
            img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"Failed to read {fn}")
            imgs.append(img.astype(np.float32))
        return np.stack(imgs, axis=-1)

    def _yolo_to_inner_outer_mask(self, ann_path, hw):
        h, w = hw
        outer = np.zeros((h, w), dtype=np.uint8)
        inner = np.zeros((h, w), dtype=np.uint8)

        with open(ann_path) as f:
            for line in f:
                toks = line.strip().split()
                if len(toks) < 3 or len(toks) % 2 == 0:
                    continue
                cls    = int(toks[0])
                coords = list(map(float, toks[1:]))
                pts = [[int(coords[i]*w), int(coords[i+1]*h)]
                       for i in range(0, len(coords), 2)]
                poly = np.array(pts, dtype=np.int32).reshape(-1,1,2)
                if cls==1:
                    cv2.fillPoly(outer, [poly], 1)
                elif cls==0:
                    cv2.fillPoly(inner, [poly], 1)

        # channel0 = inner; channel1 = outer - inner
        outer_minus_inner = np.clip(outer.astype(int) - inner.astype(int), 0, 1).astype(np.uint8)
        return np.stack([inner, outer_minus_inner], axis=-1).astype(np.float32)

    def preprocess(self, img_stack):
        """
        Linear float32 preprocessing:
          1) Median filter (3×3)
          2) Gaussian denoise (σ=1)
          3) Background subtract (Gaussian σ=50)
          4) Percentile clip (1st–99th)
          5) Rescale to [0,1]
        Returns float32 (H, W, C) in [0,1].
        """
        processed = []
        for c in range(img_stack.shape[-1]):
            ch = img_stack[..., c]
            # 1. median
            ch = median(ch, square(3))
            # 2. gaussian denoise
            ch = gaussian(ch, sigma=1.0)
            # 3. background subtract
            bg = gaussian(ch, sigma=50)
            ch = np.clip(ch - bg, 0.0, None)
            # 4. percentile clip
            p1, p99 = np.percentile(ch, (1, 99))
            if p99 > p1:  # Avoid division by zero
                ch = np.clip(ch, p1, p99)
                # 5. rescale to [0,1]
                ch = (ch - p1) / (p99 - p1)
            else:
                ch = np.zeros_like(ch)  # Handle edge case
            processed.append(ch.astype(np.float32))
        proc_stack = np.stack(processed, axis=-1)
        return proc_stack

    def gamma_correction(self, img, gamma=1.0):
        """
        Apply gamma correction to a float [0,1] image or uint8 image.
        Ensures we end up with a uint8 result in [0,255].
        """
        # if float, scale to [0,255]
        if img.dtype != np.uint8:
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255).astype(np.uint8)

        # build lookup table
        table = np.array([((i / 255.0) ** gamma) * 255
                          for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)

    def inspect_sample(self, idx=0):
        """
        Display each channel with inner (blue) & outer-only (red) overlay.
        Now uses preprocess() so even low-signal channels (FITC) show up.
        """
        # 1) load & preprocess
        img_dir, ann_path, sid = self.samples[idx]
        raw_stack = self._load_image_stack(img_dir)              # H,W,C float32
        proc_stack = self.preprocess(raw_stack)                  # H,W,C float32 in [0,1]

        # 2) get mask (inner / outer-only)
        mask = self._yolo_to_inner_outer_mask(ann_path, raw_stack.shape[:2])

        # 3) build display stack: normalize & gamma per-channel → uint8
        disp = []
        for c in range(proc_stack.shape[-1]):
            # scale to 0–255
            ch8 = (proc_stack[..., c] * 255).astype(np.uint8)
            # gamma LUT
            ch8 = self.gamma_correction(ch8, gamma=0.2)
            disp.append(ch8)
        disp = np.stack(disp, axis=-1)  # H,W,C uint8

        # 4) overlay mask (same as before)
        overlay = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
        overlay[mask[...,0]==1] = [255, 0, 0]   # inner→blue in BGR
        overlay[mask[...,1]==1] = [0, 0, 255]   # outer-only→red

        alpha = 0.5
        plt.figure(figsize=(18,6))
        for i, ch in enumerate(self.channels):
            gray = disp[..., i]
            bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            vis  = cv2.addWeighted(bgr, 1-alpha, overlay, alpha, 0)
            plt.subplot(1, len(self.channels), i+1)
            plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            plt.title(f"{sid} — {ch}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def get_debug_item(self, idx):
        img_dir, ann_path, sid = self.samples[idx]
        raw       = self._load_image_stack(img_dir)
        mask_raw  = self._yolo_to_inner_outer_mask(ann_path, raw.shape[:2]) if ann_path else None
        preproc   = self.preprocess(raw)

        if mask_raw is not None and self.transform:
            aug = self.transform(image=preproc, mask=mask_raw)
            transformed     = aug['image']
            mask_transformed = aug['mask']
        else:
            transformed     = None
            mask_transformed = None

        return {
            'sample_id': sid,
            'raw'      : raw,
            'preprocessed': preproc,
            'transformed' : transformed,
            'mask_raw'    : mask_raw,
            'mask_transformed': mask_transformed
        }


# Main training pipeline
def main():
    """Main training pipeline with all optimizations"""
    # Configuration
    config = {
        'train_data_dir': r'C:\Users\Yifei\Documents\data_for_publication\train_preprocessed',
        'val_data_dir': r'C:\Users\Yifei\Documents\data_for_publication\val_preprocessed',
        'test_data_dir': r'C:\Users\Yifei\Documents\data_for_publication\test_preprocessed',
        'output_dir': r'C:\Users\Yifei\Documents\data_for_publication\optimized_results',
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'channels': ['DAPI', 'FITC', 'TRITC'],
        'n_channels': 3,
        'n_classes': 2,
        'encoder_name': 'resnet34',
        'epochs': 30,  # For hyperparameter search
        'final_epochs': 100,  # For final training
        'patience': 15,
        'encoder_test_epochs': 20,  # Quick test for encoder comparison
        'cv_epochs': 30,  # For cross-validation
    }
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create datasets
    train_dataset = MultiChannelSegDataset(
        config['train_data_dir'], config['channels']
    )
    val_dataset = MultiChannelSegDataset(
        config['val_data_dir'], config['channels']
    )
    test_dataset = MultiChannelSegDataset(
        config['test_data_dir'], config['channels']
    )
    
    # Step 1: Compare encoder architectures
    print("\n" + "="*70)
    print("STEP 1: Comparing Encoder Architectures")
    print("="*70)
    encoder_results = compare_encoders(train_dataset, val_dataset, test_dataset, config)
    best_encoder = encoder_results.iloc[0]['encoder']
    config['encoder_name'] = best_encoder
    print(f"\nBest encoder: {best_encoder}")
    
    # Step 2: Cross-validation
    print("\n" + "="*70)
    print("STEP 2: Cross-Validation")
    print("="*70)
    # Combine train and val for CV
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    cv_results = cross_validate_model(combined_dataset, config, n_folds=5)
    
    # Step 3: Hyperparameter optimization
    print("\n" + "="*70)
    print("STEP 3: Hyperparameter Optimization")
    print("="*70)
    best_params = run_hyperparameter_search(
        train_dataset, val_dataset, config, n_trials=50
    )
    
    # Step 4: Train final model
    print("\n" + "="*70)
    print("STEP 4: Training Final Model")
    print("="*70)
    final_model, test_metrics = train_final_model(
        train_dataset, val_dataset, test_dataset, best_params, config
    )
    
    print("\n" + "="*70)
    print("Training Pipeline Complete!")
    print("="*70)
    print(f"Results saved to: {config['output_dir']}")


if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the pipeline
    main()