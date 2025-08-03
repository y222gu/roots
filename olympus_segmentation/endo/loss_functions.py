import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


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
    """Alternative loss function for handling class imbalance - Custom Implementation"""
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
    
    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten tensors
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        # Calculate Tversky components
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        
        # Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        
        # Focal Tversky loss
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky