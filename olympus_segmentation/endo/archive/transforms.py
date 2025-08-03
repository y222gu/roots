import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# --------------------- Transforms ---------------------
def get_train_transforms():
    return A.Compose([
        # 1) random crop & resize to introduce scale/position variance
        A.RandomResizedCrop(
                            size=(1024, 1024),
                            scale=(0.8, 1.0),
                            ratio=(0.9, 1.1)),

        # 2) horizontal/vertical flip
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.8, 1.2),
                rotate=(-45, 45),
                interpolation=cv2.INTER_LINEAR,
                mode=cv2.BORDER_REFLECT,
                p=0.5,
            ),

            # spatial warps — drop the old alpha_affine & shift_limit
            A.ElasticTransform(alpha=1.0, sigma=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.05, p=0.3),

            # noise & blur — use only the supported args
            A.GaussNoise(mean=0.0, per_channel=True, p=0.4),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2,
                                       p=0.5),
            A.RandomGamma(gamma_limit=(40, 80), p=0.5),

            # dropout — keep only the core params that still exist
            A.CoarseDropout(
                max_holes=5,
                max_height=64, max_width=64,
                min_holes=1,
                min_height=16, min_width=16,
                fill_value=0,
                mask_fill_value=0,
                p=0.3,
            ),

        A.Normalize(),
        ToTensorV2(),

    ],
    additional_targets={"mask": "mask"})

def get_val_transforms():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(),
        ToTensorV2(),
    ],
        additional_targets={
            "mask": "mask"
        })