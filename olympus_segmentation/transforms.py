import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------- Transforms ---------------------
def get_train_transforms():
    return A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ],
        additional_targets={
            "image_original": "image_original"
        })

def get_val_transforms():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(),
        ToTensorV2(),
    ],
        additional_targets={
            "image_original": "image_original"
        })