import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from skimage.filters import median, gaussian
from skimage.morphology import square
from transforms import get_train_transforms
import torch
import os


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
        mask  = self._yolo_to_inner_outer_mask(ann_path, image.shape[:2]) if ann_path else None

        # preprocess + augment
        # image = self.preprocess(image)
        if mask is not None:
            if self.transform:
                aug = self.transform(image=image, mask=mask)
                image, mask = aug['image'], aug['mask']

            # convert to torch.Tensor (C,H,W)        
            mask = mask.float()  

            # ***critical***: clone to force a fresh, resizable storage
            image = image.clone().contiguous()
            mask  = mask.clone().contiguous()

            return image, mask, sid
        else:
            if self.transform:
                image = self.transform(image=image)['image']
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

    data_dir = r'C:\Users\Yifei\Documents\data_for_publication\test\Zeiss'
    channels = ['DAPI', 'FITC', 'TRITC']
    transform = get_train_transforms()
    dataset = MultiChannelSegDataset(data_dir, channels, transform=transform)

    # Create output folder for figures
    output_dir = r'C:\Users\Yifei\Documents\debug_figures'
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