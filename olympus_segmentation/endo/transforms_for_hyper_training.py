import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

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

def get_org_transforms():
    """Transforms for original image (no augmentation) - FIXED"""
    return A.Compose([
        A.Resize(1024, 1024),  # This ensures all images are the same size
        ToTensorV2()
    ])

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
        A.OpticalDistortion(distort_limit=0.7, p=p_optical),  # Removed shift_limit
        
        # Intensity variations (lighter augmentation)
        # A.RandomBrightnessContrast(
        #     brightness_limit=0.1, contrast_limit=0.1, p=0.3
        # ),
        # A.RandomGamma(gamma_limit=(90, 110), p=0.2),
        # Noise and blur (simulate different imaging conditions)
        # A.GaussNoise(p=p_noise),  # Simplified - use default parameters
        # A.GaussianBlur(blur_limit=(3, 7), p=p_blur),
        
        # # Channel-wise augmentations for fluorescence
        # A.ChannelShuffle(p=0.1),  # Only occasionally
        # A.ChannelDropout(channel_drop_range=(1, 1), p=0.1),
        #         # dropout — keep only the core params that still exist
        A.CoarseDropout(
            max_holes=5,
            max_height=64, max_width=64,
            min_holes=1,
            min_height=16, min_width=16,
            fill_value=0,
            mask_fill_value=0,
            p=0.3,
        ),

    ]
    
    return A.Compose(
        base_transforms + advanced_transforms + [
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),#
            ToTensorV2()
        ],
        additional_targets={
            "mask": "mask"
        }
    )