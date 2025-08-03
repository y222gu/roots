"""
Zero-Shot Cross-Species Training Strategy for Root Segmentation
Optimized for training on single species and generalizing to others
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class DomainGeneralizationAugmentation:
    """Augmentations specifically designed to improve cross-species generalization"""
    
    def __init__(self, strength='strong'):
        self.strength = strength
        
    def get_transforms(self):
        if self.strength == 'strong':
            return A.Compose([
                # Geometric augmentations - critical for shape generalization
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.Transpose(p=0.5),
                
                # Strong elastic deformations - simulate species differences
                A.OneOf([
                    A.ElasticTransform(
                        alpha=300, sigma=15, alpha_affine=50, p=1.0
                    ),
                    A.GridDistortion(
                        num_steps=8, distort_limit=0.5, p=1.0
                    ),
                    A.OpticalDistortion(
                        distort_limit=1.0, shift_limit=0.5, p=1.0
                    ),
                ], p=0.8),
                
                # Scale variations - different root sizes across species
                A.RandomScale(
                    scale_limit=(-0.5, 1.0), p=0.7
                ),
                
                # Perspective changes - different imaging angles
                A.Perspective(
                    scale=(0.05, 0.15), p=0.5
                ),
                
                # Thickness variations - simulate species differences
                A.OneOf([
                    A.Morphological(
                        scale=(2, 5), operation='dilation', p=1.0
                    ),
                    A.Morphological(
                        scale=(2, 5), operation='erosion', p=1.0
                    ),
                ], p=0.3),
                
                # Texture/appearance variations - cross-microscope
                A.OneOf([
                    # Simulate different PSFs
                    A.GaussianBlur(blur_limit=(3, 9), p=1.0),
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                    A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.5), p=1.0),
                ], p=0.6),
                
                # Intensity variations - different staining/microscopes
                A.RandomBrightnessContrast(
                    brightness_limit=0.4, contrast_limit=0.4, p=0.8
                ),
                A.RandomGamma(
                    gamma_limit=(50, 150), p=0.7
                ),
                
                # Simulate different noise characteristics
                A.OneOf([
                    A.GaussNoise(var_limit=(10, 100), p=1.0),
                    A.MultiplicativeNoise(
                        multiplier=(0.8, 1.2), elementwise=True, p=1.0
                    ),
                ], p=0.5),
                
                # Color variations - different fluorophores
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, 
                    val_shift_limit=20, p=0.5
                ),
                
                # Simulate imaging artifacts
                A.OneOf([
                    A.ISONoise(color_shift=(0.01, 0.05), p=1.0),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=1.0),
                ], p=0.2),
                
                # Normalization
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:  # validation/test
            return A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])


class FeatureInvariantLoss(nn.Module):
    """Loss that encourages learning of invariant features"""
    
    def __init__(self, base_loss, invariance_weight=0.1):
        super().__init__()
        self.base_loss = base_loss
        self.invariance_weight = invariance_weight
        
    def forward(self, logits, targets, features=None):
        # Base segmentation loss
        seg_loss = self.base_loss(logits, targets)
        
        if features is not None and len(features) > 1:
            # Encourage similar features for augmented versions
            # This helps learn invariant representations
            invariance_loss = 0
            for i in range(1, len(features)):
                invariance_loss += F.mse_loss(features[0], features[i])
            invariance_loss /= (len(features) - 1)
            
            return seg_loss + self.invariance_weight * invariance_loss
        
        return seg_loss


class GeneralizableUNet(nn.Module):
    """UNet with modifications for better generalization"""
    
    def __init__(self, encoder_name='efficientnet-b4', in_channels=3, classes=2):
        super().__init__()
        
        # Base model
        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=classes
        )
        
        # Additional regularization
        self.dropout = nn.Dropout2d(0.2)
        
        # Style normalization layers for domain adaptation
        self.style_norm = nn.InstanceNorm2d(classes, affine=True)
        
    def forward(self, x, return_features=False):
        # Get encoder features
        features = self.base_model.encoder(x)
        
        # Decoder with dropout
        decoder_output = self.base_model.decoder(*features)
        decoder_output = self.dropout(decoder_output)
        
        # Final segmentation
        masks = self.base_model.segmentation_head(decoder_output)
        
        # Apply style normalization for better generalization
        masks = self.style_norm(masks)
        
        if return_features:
            # Return intermediate features for invariance loss
            return masks, features[-1]  # Last encoder feature
        
        return masks


class TestTimeAugmentation:
    """TTA specifically for cross-species inference"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def predict(self, image):
        """Predict with test-time augmentation"""
        transforms = [
            lambda x: x,  # Original
            lambda x: torch.flip(x, [2]),  # Horizontal flip
            lambda x: torch.flip(x, [3]),  # Vertical flip
            lambda x: torch.rot90(x, 1, [2, 3]),  # 90 degree rotation
            lambda x: torch.rot90(x, 2, [2, 3]),  # 180 degree rotation
            lambda x: torch.rot90(x, 3, [2, 3]),  # 270 degree rotation
        ]
        
        inverse_transforms = [
            lambda x: x,  # Original
            lambda x: torch.flip(x, [2]),  # Horizontal flip (self-inverse)
            lambda x: torch.flip(x, [3]),  # Vertical flip (self-inverse)
            lambda x: torch.rot90(x, -1, [2, 3]),  # -90 degree rotation
            lambda x: torch.rot90(x, -2, [2, 3]),  # -180 degree rotation
            lambda x: torch.rot90(x, -3, [2, 3]),  # -270 degree rotation
        ]
        
        predictions = []
        
        with torch.no_grad():
            for transform, inverse in zip(transforms, inverse_transforms):
                # Apply transform
                augmented = transform(image)
                
                # Predict
                pred = self.model(augmented)
                
                # Apply inverse transform
                pred = inverse(pred)
                
                predictions.append(torch.sigmoid(pred))
        
        # Average predictions
        return torch.mean(torch.stack(predictions), dim=0)


def train_for_generalization(model, train_loader, val_loader, config):
    """Training loop optimized for cross-species generalization"""
    
    # Loss and optimizer
    base_loss = smp.losses.DiceLoss(mode='multilabel')
    loss_fn = FeatureInvariantLoss(base_loss, invariance_weight=0.1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    # Cosine annealing with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    best_val_dice = 0.0
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, (imgs, masks, _) in enumerate(train_loader):
            imgs = imgs.to(config['device'])
            masks = masks.to(config['device']).permute(0, 3, 1, 2)
            
            # Create multiple augmented versions for invariance
            batch_size = imgs.size(0)
            augmented_imgs = [imgs]
            augmented_masks = [masks]
            
            # Add strongly augmented versions
            for _ in range(2):  # 2 additional augmented versions
                # Apply same augmentation to image and mask
                aug = A.Compose([
                    A.ElasticTransform(alpha=200, sigma=10, p=1.0),
                    A.RandomBrightnessContrast(p=1.0),
                ])
                
                aug_imgs = []
                aug_masks = []
                for i in range(batch_size):
                    augmented = aug(
                        image=imgs[i].cpu().numpy().transpose(1, 2, 0),
                        mask=masks[i].cpu().numpy().transpose(1, 2, 0)
                    )
                    aug_imgs.append(
                        torch.tensor(augmented['image']).permute(2, 0, 1)
                    )
                    aug_masks.append(
                        torch.tensor(augmented['mask']).permute(2, 0, 1)
                    )
                
                augmented_imgs.append(torch.stack(aug_imgs).to(config['device']))
                augmented_masks.append(torch.stack(aug_masks).to(config['device']))
            
            # Forward pass with all versions
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                all_logits = []
                all_features = []
                
                for aug_imgs, aug_masks in zip(augmented_imgs, augmented_masks):
                    logits, features = model(aug_imgs, return_features=True)
                    all_logits.append(logits)
                    all_features.append(features)
                
                # Compute loss with invariance term
                loss = 0
                for logits, masks in zip(all_logits, augmented_masks):
                    loss += loss_fn(logits, masks, all_features)
                loss /= len(all_logits)
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            # Step scheduler
            scheduler.step(epoch + batch_idx / len(train_loader))
        
        # Validation
        model.eval()
        val_dice = validate_model(model, val_loader, config['device'])
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Dice: {val_dice:.4f}")
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best_model_generalization.pth')
    
    return model


def validate_model(model, loader, device):
    """Validation with proper metrics"""
    model.eval()
    dice_scores = []
    
    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs = imgs.to(device)
            masks = masks.to(device).permute(0, 3, 1, 2)
            
            logits = model(imgs)
            probs = torch.sigmoid(logits)
            
            # Compute dice
            preds = (probs > 0.5).float()
            intersection = (preds * masks).sum(dim=(2, 3))
            union = preds.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            dice_scores.extend(dice.mean(dim=1).cpu().numpy())
    
    return np.mean(dice_scores)


# Additional strategies for zero-shot generalization

class MixStyleAugmentation:
    """Mix styles between different augmented versions to simulate domain shift"""
    
    @staticmethod
    def apply(images, alpha=0.1):
        batch_size = images.size(0)
        
        # Compute instance statistics
        mean = images.mean(dim=[2, 3], keepdim=True)
        std = images.std(dim=[2, 3], keepdim=True)
        
        # Shuffle statistics within batch
        perm = torch.randperm(batch_size)
        mean_perm = mean[perm]
        std_perm = std[perm]
        
        # Mix statistics
        mixed_mean = (1 - alpha) * mean + alpha * mean_perm
        mixed_std = (1 - alpha) * std + alpha * std_perm
        
        # Apply new statistics
        normalized = (images - mean) / (std + 1e-6)
        mixed = normalized * mixed_std + mixed_mean
        
        return mixed


class ConsistencyRegularization:
    """Enforce prediction consistency across augmentations"""
    
    def __init__(self, consistency_weight=1.0):
        self.consistency_weight = consistency_weight
        
    def compute_loss(self, predictions_list):
        """Compute consistency loss between predictions"""
        consistency_loss = 0
        n_predictions = len(predictions_list)
        
        if n_predictions > 1:
            # Use first prediction as reference
            reference = predictions_list[0].detach()
            
            for i in range(1, n_predictions):
                consistency_loss += F.mse_loss(predictions_list[i], reference)
            
            consistency_loss /= (n_predictions - 1)
        
        return self.consistency_weight * consistency_loss


# Main training configuration for zero-shot generalization
def get_zero_shot_config():
    return {
        'augmentation': DomainGeneralizationAugmentation(strength='strong'),
        'model': GeneralizableUNet(encoder_name='efficientnet-b4'),
        'loss': FeatureInvariantLoss(
            smp.losses.DiceLoss(mode='multilabel'),
            invariance_weight=0.1
        ),
        'lr': 3e-4,
        'weight_decay': 5e-5,
        'epochs': 150,  # More epochs needed for harder task
        'use_mixstyle': True,
        'use_consistency': True,
        'tta_inference': True,  # Use test-time augmentation
    }