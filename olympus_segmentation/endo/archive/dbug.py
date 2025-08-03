import os
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from skimage.filters import median, gaussian
from skimage.morphology import square
import cv2
import matplotlib.pyplot as plt


# --------------------- Dataset Class ---------------------
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
        mask = self._yolo_to_inner_outer_mask(ann_path, image.shape[:2]) if ann_path else None

        if mask is not None:
            if self.transform:
                aug = self.transform(image=image, mask=mask)
                image, mask = aug['image'], aug['mask']

            # Convert numpy arrays to torch tensors FIRST, then call .float()
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).permute(2, 0, 1)  # H,W,C -> C,H,W
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).permute(2, 0, 1)    # H,W,C -> C,H,W
            else:
                # If mask is already a tensor but has wrong dimensions, fix it
                if len(mask.shape) == 3 and mask.shape[0] != 2:  # H,W,C format
                    mask = mask.permute(2, 0, 1)  # H,W,C -> C,H,W
            
            # Ensure image has correct dimensions
            if len(image.shape) == 3 and image.shape[0] != 3:  # H,W,C format
                image = image.permute(2, 0, 1)  # H,W,C -> C,H,W
            
            # Now we can safely call .float() since they are tensors
            image = image.float()
            mask = mask.float()

            # ***critical***: clone to force a fresh, resizable storage
            image = image.clone().contiguous()
            mask = mask.clone().contiguous()

            return image, mask, sid
        else:
            if self.transform:
                image = self.transform(image=image)['image']

            # Convert to torch.Tensor if not already
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).permute(2, 0, 1)  # H,W,C -> C,H,W
            else:
                # If image is already a tensor but has wrong dimensions, fix it
                if len(image.shape) == 3 and image.shape[0] != 3:  # H,W,C format
                    image = image.permute(2, 0, 1)  # H,W,C -> C,H,W
            
            image = image.float()
            # ***critical***: clone to force a fresh, resizable storage
            image = image.clone().contiguous()

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
            # img = cv2.imread(os.path.join(folder, fn), cv2.IMREAD_GRAYSCALE)
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
        return np.stack([inner, outer_minus_inner], axis=-1)

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
            ch = np.clip(ch, p1, p99)
            # 5. rescale to [0,1]
            ch = (ch - p1) / (p99 - p1)
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
def test_tensor_dimensions():
    """Test that all tensors have consistent dimensions"""

    
    # Configuration
    data_dir = r'C:\Users\Yifei\Documents\data_for_publication\train_preprocessed'
    channels = ['DAPI', 'FITC', 'TRITC']
    
    print("Testing tensor dimensions...")
    
    dataset = MultiChannelSegDataset(data_dir, channels)
    
    # Create transform with resize
    transform = A.Compose([
        A.Resize(1024, 1024, always_apply=True),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})
    
    dataset.transform = transform
    
    print("\nChecking individual samples:")
    all_good = True
    
    for i in range(min(5, len(dataset))):
        try:
            sample = dataset[i]
            if len(sample) == 3:
                image, mask, sid = sample
                print(f"Sample {i} ({sid}):")
                print(f"  Image: {image.shape} (should be [3, 1024, 1024])")
                print(f"  Mask:  {mask.shape} (should be [2, 1024, 1024])")
                
                # Check dimensions
                if image.shape != torch.Size([3, 1024, 1024]):
                    print(f"  ❌ Image shape incorrect!")
                    all_good = False
                if mask.shape != torch.Size([2, 1024, 1024]):
                    print(f"  ❌ Mask shape incorrect!")
                    all_good = False
                if all_good:
                    print(f"  ✅ Dimensions correct")
                    
            else:
                image, sid = sample
                print(f"Sample {i} ({sid}) - no mask:")
                print(f"  Image: {image.shape} (should be [3, 1024, 1024])")
                
                if image.shape != torch.Size([3, 1024, 1024]):
                    print(f"  ❌ Image shape incorrect!")
                    all_good = False
                else:
                    print(f"  ✅ Dimensions correct")
                    
        except Exception as e:
            print(f"Sample {i} failed: {e}")
            all_good = False
    
    if all_good:
        print("\n✅ All individual samples have correct dimensions")
    else:
        print("\n❌ Some samples have incorrect dimensions")
        return False
    
    # Test DataLoader
    print("\nTesting DataLoader with different batch sizes:")
    
    for batch_size in [1, 2, 4]:
        try:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            batch = next(iter(dataloader))
            
            if len(batch) == 3:
                images, masks, sids = batch
                print(f"Batch size {batch_size}:")
                print(f"  Images: {images.shape} (should be [{batch_size}, 3, 1024, 1024])")
                print(f"  Masks:  {masks.shape} (should be [{batch_size}, 2, 1024, 1024])")
                
                expected_img_shape = torch.Size([batch_size, 3, 1024, 1024])
                expected_mask_shape = torch.Size([batch_size, 2, 1024, 1024])
                
                if images.shape == expected_img_shape and masks.shape == expected_mask_shape:
                    print(f"  ✅ Batch dimensions correct")
                else:
                    print(f"  ❌ Batch dimensions incorrect!")
                    all_good = False
                    
            else:
                images, sids = batch
                print(f"Batch size {batch_size} (no masks):")
                print(f"  Images: {images.shape} (should be [{batch_size}, 3, 1024, 1024])")
                
                expected_img_shape = torch.Size([batch_size, 3, 1024, 1024])
                if images.shape == expected_img_shape:
                    print(f"  ✅ Batch dimensions correct")
                else:
                    print(f"  ❌ Batch dimensions incorrect!")
                    all_good = False
                    
        except Exception as e:
            print(f"Batch size {batch_size} failed: {e}")
            import traceback
            traceback.print_exc()
            all_good = False
    
    if all_good:
        print("\n🎉 All dimension tests passed!")
        return True
    else:
        print("\n❌ Some dimension tests failed!")
        return False

def test_with_your_training_transforms():
    """Test with the actual transforms from your training code"""
    
    try:
        from endo_dataset import MultiChannelSegDataset
    except ImportError:
        print("Cannot import dataset")
        return
    
    print("\n" + "="*50)
    print("Testing with your actual training transforms")
    print("="*50)
    
    # Your actual transforms from the training code
    train_transform = A.Compose([
        A.RandomResizedCrop(size=(1024, 1024), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})
    
    val_transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], additional_targets={"mask": "mask"})
    
    data_dir = r'C:\Users\Yifei\Documents\data_for_publication\train_preprocessed'
    channels = ['DAPI', 'FITC', 'TRITC']
    
    dataset = MultiChannelSegDataset(data_dir, channels)
    
    # Test training transforms
    print("\nTesting training transforms:")
    dataset.transform = train_transform
    
    for i in range(3):
        try:
            sample = dataset[i]
            if len(sample) == 3:
                image, mask, sid = sample
                print(f"  Sample {i} ({sid}): Image {image.shape}, Mask {mask.shape}")
            else:
                image, sid = sample
                print(f"  Sample {i} ({sid}): Image {image.shape}")
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
    
    # Test validation transforms
    print("\nTesting validation transforms:")
    dataset.transform = val_transform
    
    for i in range(3):
        try:
            sample = dataset[i]
            if len(sample) == 3:
                image, mask, sid = sample
                print(f"  Sample {i} ({sid}): Image {image.shape}, Mask {mask.shape}")
            else:
                image, sid = sample
                print(f"  Sample {i} ({sid}): Image {image.shape}")
        except Exception as e:
            print(f"  Sample {i} failed: {e}")
    
    # Test DataLoader with validation transforms
    print("\nTesting DataLoader with validation transforms:")
    try:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        
        if len(batch) == 3:
            images, masks, sids = batch
            print(f"  Batch: Images {images.shape}, Masks {masks.shape}")
        else:
            images, sids = batch
            print(f"  Batch: Images {images.shape}")
            
        print("  ✅ DataLoader test passed!")
        
    except Exception as e:
        print(f"  ❌ DataLoader test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    success = test_tensor_dimensions()
    if success:
        test_with_your_training_transforms()