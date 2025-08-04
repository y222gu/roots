import numpy as np
import os
import torch
import segmentation_models_pytorch as smp
from tqdm import tqdm
import cv2
from torch.utils.data import Dataset
import gc
import time
from matplotlib import pyplot as plt
from transforms_for_hyper_training import get_val_transforms, get_augmented_transforms


class ModelTrainer:
    """Enhanced trainer with advanced features"""
    def __init__(self, model, device, loss_fn, optimizer, scheduler=None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Fix for torch.amp.GradScaler - use torch.amp.GradScaler instead of deprecated version
        self.scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    def train_epoch(self, loader, epoch_num=None):
        """Train for one epoch with mixed precision"""
        self.model.train()
        total_loss = 0.0
        metric_tracker = MetricTracker()
        valid_batches = 0
        
        pbar = tqdm(loader, desc=f'Training epoch {epoch_num}')
        for batch_idx, batch in enumerate(pbar):
            # Handle both cases: with and without masks
            if len(batch) == 3:
                imgs, masks, _ = batch
                has_masks = True
            else:
                imgs, _ = batch
                has_masks = False
                continue  # Skip samples without masks during training
            
            # CRITICAL FIX: Skip batches with size 1 to avoid BatchNorm issues
            if imgs.size(0) == 1:
                print(f"Skipping batch {batch_idx} with size 1 to avoid BatchNorm issues")
                continue
                       
            imgs = imgs.to(self.device, non_blocking=True)  # non_blocking for faster transfer
            if has_masks:
                masks = masks.to(self.device, non_blocking=True)  # non_blocking for faster transfer
                                    # Check if masks need permutation (H,W,C -> C,H,W)
                # if masks.dim() == 4 and masks.shape[-1] == 2:  # B,H,W,C
                #     masks = masks.permute(0, 3, 1, 2)  # B,C,H,W

            self.optimizer.zero_grad()
            
            # Mixed precision forward pass (only if CUDA and scaler available)
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):  # Updated autocast syntax
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
            else:
                # Standard training without mixed precision
                logits = self.model(imgs)
                if has_masks:
                    loss = self.loss_fn(logits, masks)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    self.optimizer.step()
            
            # Update metrics
            if has_masks:
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    metric_tracker.update(probs, masks)
                
                total_loss += loss.item() * imgs.size(0)
                valid_batches += 1
                pbar.set_postfix({'loss': loss.item()})
        
        if self.scheduler:
            self.scheduler.step()
        
        # Calculate average loss over valid batches only
        avg_loss = total_loss / (valid_batches * loader.batch_size) if valid_batches > 0 else 0
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
                
                # CRITICAL FIX: Skip batches with size 1 to avoid BatchNorm issues
                if imgs.size(0) == 1:
                    print(f"Skipping validation batch with size 1")
                    continue
                
                imgs = imgs.to(self.device)
                if has_masks:
                    masks = masks.to(self.device)
                    # Check if masks need permutation (H,W,C -> C,H,W)
                    # if masks.dim() == 4 and masks.shape[-1] == 2:  # B,H,W,C
                    #     masks = masks.permute(0, 3, 1, 2)  # B,C,H,W
                
                # Use mixed precision if available, otherwise standard forward pass
                if self.scaler is not None:
                    with torch.amp.autocast('cuda'):  # Updated autocast syntax
                        logits = self.model(imgs)
                        if has_masks:
                            loss = self.loss_fn(logits, masks)
                else:
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
        

def create_model(encoder_name, architecture, n_channels, n_classes):
    if architecture == 'Unet':
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=n_channels,
            classes=n_classes
        )
    elif architecture == 'UnetPlusPlus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=n_channels,
            classes=n_classes
        )
    elif architecture == 'DeepLabV3Plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=n_channels
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    return model



class MultiChannelSegDataset(Dataset):
    def __init__(self, data_dir, channels, transform=None, manual_annotation=True):
        """
        data_dir: root directory (e.g. "Training/")
        channels: list of substrings to identify each channel file (['DAPI','FITC','TRITC'])
        """
        self.data_dir = data_dir  # Store data_dir as an attribute
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
        try:
            img_dir, ann_path, sid = self.samples[idx]
            image = self._load_image_stack(img_dir)         #  CxH×W float32

            mask = self._yolo_to_inner_outer_mask(ann_path, image.shape[:2]) if ann_path else None

            if mask is not None:
                # Apply transforms if provided
                if self.transform:
                    aug = self.transform(image=image, mask=mask)
                    image, mask = aug['image'], aug['mask']
                    mask = mask.permute(2, 0, 1)  # B,C,H,W

                    # Ensure proper tensor format and contiguity
                    if isinstance(image, torch.Tensor):
                        image = image.clone().detach().contiguous()
                    if isinstance(mask, torch.Tensor):
                        mask = mask.clone().detach().contiguous()

                else:
                    # If no transforms, manually resize and convert to tensor
                    # Resize to consistent size
                    image_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    mask_resized = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)
                    
                    # Normalize image
                    image_resized = image_resized / 255.0  # Normalize to [0,1]
                    # Apply ImageNet normalization
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image_resized = (image_resized - mean) / std
                    
                    # Convert to tensors
                    image = torch.from_numpy(image_resized.copy()).permute(2, 0, 1).float().contiguous()  # H,W,C -> C,H,W
                    mask = torch.from_numpy(mask_resized.copy()).permute(2, 0, 1).float().contiguous()    # H,W,C -> C,H,W

                return image, mask, sid
            else:
                # Handle case with no mask
                if self.transform:
                    # Create a dummy mask for augmentation (will be ignored)
                    dummy_mask = np.zeros((*image.shape[:2], 2), dtype=np.float32)
                    aug = self.transform(image=image, mask=dummy_mask)
                    image = aug['image']
                    
                    # Ensure proper tensor format and contiguity
                    if isinstance(image, torch.Tensor):
                        image = image.clone().detach().contiguous()
                else:
                    # Convert to tensor manually with consistent size
                    image_resized = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)
                    image_resized = image_resized / 255.0  # Normalize to [0,1]
                    # Apply ImageNet normalization
                    mean = np.array([0.485, 0.456, 0.406])
                    std = np.array([0.229, 0.224, 0.225])
                    image_resized = (image_resized - mean) / std
                    image = torch.from_numpy(image_resized.copy()).permute(2, 0, 1).float().contiguous()

                # Verify tensor shape is correct
                assert image.shape[1:] == (1024, 1024), f"Image shape mismatch: {image.shape}"

                return image, sid
                
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None

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
            
            # Normalize to [0, 255] range if needed
            if img.dtype == np.uint16:
                # Convert 16-bit to 8-bit by scaling
                img = (img / 256).astype(np.uint8)
            elif img.dtype != np.uint8:
                # Scale to 0-255 range for other dtypes
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
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
        mask = np.stack([inner, outer_minus_inner], axis=-1).astype(np.float32)  # [H, W, C]
        return mask  # Now returns [H, W, C]
    
    def get_debug_item(self, idx):
        img_dir, ann_path, sid = self.samples[idx]
        raw       = self._load_image_stack(img_dir)
        mask_raw  = self._yolo_to_inner_outer_mask(ann_path, raw.shape[:2]) if ann_path else None
        # preproc   = self.preprocess(raw)

        if mask_raw is not None and self.transform:
            aug = self.transform(image=raw, mask=mask_raw)
            transformed     = aug['image']
            mask_transformed = aug['mask']
        else:
            transformed     = None
            mask_transformed = None

        return {
            'sample_id': sid,
            'raw'      : raw,
            'preprocessed': raw,
            'transformed' : transformed,
            'mask_raw'    : mask_raw,
            'mask_transformed': mask_transformed
        }

def aggressive_cleanup():
    """More aggressive memory cleanup"""
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # Small delay to allow cleanup
    time.sleep(0.5)

def cleanup_memory():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# Additional utility function for memory monitoring
def monitor_gpu_memory():
    """Monitor and print GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 512**3  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / 512**3   # GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / 512**3  # GB
            print(f"GPU {i}: {memory_allocated:.1f}GB/{memory_total:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
    else:
        print("CUDA not available")



def visualize_augmented(img, ax, channel_idx, mean=None, std=None):
    """
    img:   either a torch.Tensor (C,H,W) or a numpy array (H,W,C) in [0,1]
    ax:    matplotlib Axes
    channel_idx: which channel to display
    mean,std: optional arrays to un-normalize [only used for torch.Tensor]
    """
    # 1) convert to H×W×C numpy in [0,1]
    if isinstance(img, torch.Tensor):
        # Tensor: C×H×W -> H×W×C
        arr = img.permute(1,2,0).cpu().numpy()
        if (mean is not None) and (std is not None):
            # unnormalize
            arr = arr * std[None,None,:] + mean[None,None,:]
        arr = np.clip(arr, 0, 1)
    else:
        # assume numpy H×W×C already in [0,1]
        arr = img

    # 2) grab the requested channel and plot
    channel = arr[..., channel_idx]
    ax.imshow(channel, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


if __name__ == '__main__':

    data_dir = r'C:\Users\yifei\Documents\data_for_publication\test_preprocessed\C10\Sorghum'
    channels = ['DAPI', 'FITC', 'TRITC']
    transform = get_augmented_transforms()
    dataset = MultiChannelSegDataset(data_dir, channels, transform=transform)

    # Create output folder for figures
    output_dir = r'C:\Users\yifei\Documents\debug_figures'
    os.makedirs(output_dir, exist_ok=True)

    # Process all samples in the dataset and save figures with their sample id as filename
    for idx in range(len(dataset)):
        dbg = dataset.get_debug_item(idx)
        sample_id = dbg['sample_id']
        print("Processing Sample:", sample_id)
        print(" raw shape:", dbg['raw'].shape, dbg['raw'].dtype)
        print(" prep shape:", dbg['preprocessed'].shape, dbg['preprocessed'].dtype)
        if dbg['transformed'] is not None:
            print(" aug shape:", dbg['transformed'].shape, dbg['transformed'].dtype)

        n_ch = len(channels)
        mean = np.array([0.5]*n_ch)
        std  = np.array([0.5]*n_ch)

        # Build color overlays for raw and augmented masks
        mask_raw = dbg['mask_raw']  # H×W×2 uint8
        mask_aug = dbg['mask_transformed']  # H×W×2 uint8
        mask_raw_overlay = np.zeros((*mask_raw.shape[:2], 3), dtype=np.uint8)
        mask_raw_overlay[mask_raw[...,0]==1] = [255, 0, 0]   # blue for inner (BGR)
        mask_raw_overlay[mask_raw[...,1]==1] = [0,   0, 255]   # red for outer-only

        if mask_aug is not None:
            mask_aug_overlay = np.zeros((*mask_aug.shape[:2], 3), dtype=np.uint8)
            mask_aug_overlay[mask_aug[...,0]==1] = [255, 0, 0]   # blue for inner (BGR)
            mask_aug_overlay[mask_aug[...,1]==1] = [0,   0, 255]   # red for outer-only

        alpha = 0.5

        # Create figure with 3 rows for RAW, PREP, and AUG views
        fig, axes = plt.subplots(3, n_ch, figsize=(4*n_ch, 12))
        for ch_idx, ch in enumerate(dataset.channels):
            # RAW view with overlay
            axes[0, ch_idx].imshow(dbg['raw'][..., ch_idx], cmap='gray')
            axes[0, ch_idx].imshow(mask_raw_overlay, alpha=alpha)
            axes[0, ch_idx].set_title(f"RAW {ch}")
            axes[0, ch_idx].axis('off')

            # Preprocessed view
            axes[1, ch_idx].imshow(dbg['preprocessed'][..., ch_idx], cmap='gray')
            axes[1, ch_idx].set_title(f"PREP {ch}")
            axes[1, ch_idx].axis('off')

            # Transformed view with overlay if available
            if dbg['transformed'] is not None:
                img_t = dbg['transformed']        # torch.Tensor (C,H,W)
                visualize_augmented(img_t, axes[2, ch_idx], ch_idx, mean, std)
                axes[2, ch_idx].set_title(f"AUG {ch}")
                if mask_aug is not None:
                    axes[2, ch_idx].imshow(mask_aug_overlay, alpha=alpha)
                axes[2, ch_idx].axis('off')
            else:
                axes[2, ch_idx].text(0.5, 0.5, 'no transform', ha='center', va='center')
                axes[2, ch_idx].axis('off')

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{sample_id}.png")
        plt.savefig(out_path)
        plt.close(fig)

    print("Debug item inspection complete. Figures saved to", output_dir)