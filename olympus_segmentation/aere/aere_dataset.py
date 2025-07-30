import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transforms import get_train_transforms, get_val_transforms
import torch


# --------------------- Dataset Class ---------------------
class BinarySegDataset(Dataset):
    def __init__(self, data_dir, channels, transform=None, manual_annotation = 'True'):
        """
        Args:
            image_dir (str): Path to the main data folder that contains three subfolders: DAPI, GFP, TRITC.
            annotation_dir (str): Path to the folder with annotation files.
            transform: Optional albumentations transform to be applied on the images and masks.
        """
        self.image_dir = os.path.join(data_dir, 'image') # Assuming images are in the same folder as annotations
        self.manual_annotation = manual_annotation
        if manual_annotation == 'True':
            self.annotation_dir = os.path.join(data_dir, 'annotation')
        else:
            self.annotation_dir = None
        self.transform = transform
        self.channels = channels

        # Get the all subfolder names in the image directory as sample id.
        self.sample_ids = [f for f in os.listdir(self.image_dir ) if os.path.isdir(os.path.join(self.image_dir , f))]

        # loaf all the image stacks
        self.data = []
        if manual_annotation == 'True':
            for sample_id in self.sample_ids:
                image_stack = self.load_an_image_stack(sample_id)
                annotation = self.yolo_polygon_to_mask(sample_id, image_stack)
                if image_stack is None or annotation is None:
                    print(f"Skipping sample {sample_id} due to missing data.")
                    continue
                self.data.append((image_stack, annotation, sample_id))

        else:
            for sample_id in self.sample_ids:
                image_stack = self.load_an_image_stack(sample_id)
                if image_stack is None:
                    print(f"Skipping sample {sample_id} due to missing data.")
                    continue
                self.data.append((image_stack, sample_id))
        
        print(f"Found {len(self.data)} samples")
        # print all sample ids of data
        print("Sample IDs:", [d[2] for d in self.data] if manual_annotation == 'True' else [d[1] for d in self.data])


    def load_an_image_stack(self, sample_id):
        """
        Load images from three channels (DAPI, GFP, TRITC) for a given sample ID.
        Returns a stacked image of shape (height, width, 3).
        """
        # load the images from the subfolders with the sample_id.
        image_subfolder = os.path.join(self.image_dir, sample_id)
        if not os.path.exists(image_subfolder):
            raise FileNotFoundError(f"Image directory not found: {image_subfolder}")
        
        # List of channels to load.
        imgs = []
        # For each channel, search for a file that contains the channel name.
        for ch in self.channels:
            found_file = None
            for file in os.listdir(image_subfolder):
                if ch in file:
                    found_file = file
                    break
            if not found_file:
                return None  # No file found for this channel
            img_path = os.path.join(image_subfolder, found_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None  # Image could not be read
            imgs.append(img)
        return np.stack(imgs, axis=-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        if self.manual_annotation == 'True':
            image_original, mask, sample_id = data

            # preprocess
            image = self.gamma_correction(image_original, gamma=0.2)
            image = image.astype(np.uint8)
            image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            image = cv2.GaussianBlur(image, (5, 5), 0)

            # to tensor
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            return image, mask, sample_id, image_original
        
        else:
            image_original, sample_id = data

            # preprocess
            image = self.gamma_correction(image_original, gamma=0.2)
            image = image.astype(np.uint8)
            image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            image = cv2.GaussianBlur(image, (5, 5), 0)
            # to tensor
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, sample_id, image_original

    def yolo_polygon_to_mask(self, sample_id, image):
        """
        Convert YOLO polygon annotations for a “donut” (outer + inner ring)
        into a binary segmentation mask:
            0 = background (outside outer OR inside inner)
            1 = donut region (between outer and inner)
        Expects each line in the label file to be:
            <class> <x1> <y1> <x2> <y2> ... <xn> <yn>
        where class 0 = outer boundary, class 1 = inner boundary.
        """

        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # find the label file for this sample
        label_path = None
        for fn in os.listdir(self.annotation_dir):
            if fn.endswith('.txt') and os.path.splitext(fn)[0] == sample_id:
                label_path = os.path.join(self.annotation_dir, fn)
                break
        if label_path is None:
            return None
        
        aere_polys = []

        with open(label_path, 'r') as f:
            for line in f:
                toks = line.strip().split()
                # must have at least class + one point (i.e. 3 tokens) and odd count
                if len(toks) < 3 or len(toks) % 2 == 0:
                    continue
                cls = int(toks[0])
                coords = list(map(float, toks[1:]))
                pts = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    pts.append([x, y])
                poly = np.array(pts, dtype=np.int32).reshape((-1,1,2))

                if cls ==  1:
                    aere_polys.append(poly)
                # 0 is aere, 1 is whole root


        # Then “erase” the inner region (make it background again)
        if aere_polys:
            cv2.fillPoly(mask, aere_polys, color=1)

        # # plot mask for debugging
        # plt.figure(figsize=(6, 6))
        # plt.imshow(mask, cmap='gray')
        # plt.title(f"Mask for sample {sample_id}")
        # plt.axis('off')
        # plt.show()
        return mask

    def gamma_correction(self, image, gamma=1.0):

        # build lookup table [0..255] → [0..255]^(1/gamma)
        table = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")

        img_norm = cv2.LUT(image, table)
        return img_norm

    def inspect_sample(self, idx=0):
        """
        Load and display a sample and its segmentation mask overlay on each individual channel.
        For each channel (DAPI, GFP, TRITC), this method displays the grayscale image with the segmentation
        overlay applied (red for outer ring, blue for inner ring).
        """
        image, mask, sample_id = self.data[idx]

        # gamma correction for better visualization
        img_norm = self.gamma_correction(image, gamma=0.2)

        # normalize the mask
        img_norm = img_norm.astype(np.uint8)  # Ensure mask is in uint8 format
        img_norm = cv2.normalize(img_norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # smooth the image to reduce noise
        img_norm = cv2.GaussianBlur(img_norm, (5, 5), 0)

        # ensure mask has 3 channels for overlay
        # plot the normalized image
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        # unique_vals = np.unique(mask)
        # print("Unique values in mask:", unique_vals)

        overlay_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        overlay_mask[mask == 1] = [0, 0, 255]
        alpha = 0.5

        # loop through each layer in the image stack.
        channels = ['DAPI', 'FITC', 'TRITC']
        imgs = [img_norm[:, :, i] for i in range(img_norm.shape[2])]
        
        plt.figure(figsize=(18, 6))
        for i, ch in enumerate(channels):
            channel_img = imgs[i]
            channel_img_color = cv2.cvtColor(channel_img, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(channel_img_color, 1 - alpha, overlay_mask, alpha, 0)
            plt.subplot(1, 3, i + 1)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f"sample_{sample_id}_{ch}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()



# --------------------- Dataset Class ---------------------
class MultiLabelSegDataset(Dataset):
    def __init__(self, data_dir, channels, transform=None, manual_annotation = 'True'):
        """
        Args:
            image_dir (str): Path to the main data folder that contains three subfolders: DAPI, GFP, TRITC.
            annotation_dir (str): Path to the folder with annotation files.
            transform: Optional albumentations transform to be applied on the images and masks.
        """
        self.image_dir = os.path.join(data_dir, 'image') # Assuming images are in the same folder as annotations
        self.manual_annotation = manual_annotation
        if manual_annotation == 'True':
            self.annotation_dir = os.path.join(data_dir, 'annotation')
        else:
            self.annotation_dir = None
        self.transform = transform
        self.channels = channels

        # Get the all subfolder names in the image directory as sample id.
        self.sample_ids = [f for f in os.listdir(self.image_dir ) if os.path.isdir(os.path.join(self.image_dir , f))]

        # loaf all the image stacks
        self.data = []
        if manual_annotation == 'True':
            for sample_id in self.sample_ids:
                image_stack = self.load_an_image_stack(sample_id)
                annotation = self.yolo_polygon_to_mask(sample_id, image_stack)
                if image_stack is None or annotation is None:
                    print(f"Skipping sample {sample_id} due to missing data.")
                    continue
                self.data.append((image_stack, annotation, sample_id))

        else:
            for sample_id in self.sample_ids:
                image_stack = self.load_an_image_stack(sample_id)
                if image_stack is None:
                    print(f"Skipping sample {sample_id} due to missing data.")
                    continue
                self.data.append((image_stack, sample_id))
        
        print(f"Found {len(self.data)} samples")
        # print all sample ids of data
        print("Sample IDs:", [d[2] for d in self.data] if manual_annotation == 'True' else [d[1] for d in self.data])


    def load_an_image_stack(self, sample_id):
        """
        Load images from three channels (DAPI, GFP, TRITC) for a given sample ID.
        Returns a stacked image of shape (height, width, 3).
        """
        # load the images from the subfolders with the sample_id.
        image_subfolder = os.path.join(self.image_dir, sample_id)
        if not os.path.exists(image_subfolder):
            raise FileNotFoundError(f"Image directory not found: {image_subfolder}")
        
        # List of channels to load.
        imgs = []
        # For each channel, search for a file that contains the channel name.
        for ch in self.channels:
            found_file = None
            for file in os.listdir(image_subfolder):
                if ch in file:
                    found_file = file
                    break
            if not found_file:
                return None  # No file found for this channel
            img_path = os.path.join(image_subfolder, found_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None  # Image could not be read
            imgs.append(img)
        return np.stack(imgs, axis=-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            if self.manual_annotation == 'True':
                img_np, mask_np, sample_id = self.data[idx]
                # preprocess image
                img = self.gamma_correction(img_np, gamma=0.2)
                img = img.astype(np.uint8)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = cv2.GaussianBlur(img, (5,5), 0)

                aug = self.transform(image=img, mask=mask_np)
                image_tensor = aug['image']       # (3, H, W)
                mask_tensor  = aug['mask'] # (2, H, W)

                if isinstance(mask_tensor, np.ndarray):
                    mask_tensor = torch.from_numpy(mask_tensor)
                mask_tensor = mask_tensor.permute(2, 0, 1).float()  # [2, H, W]
                
                return image_tensor, mask_tensor, sample_id, img_np

            else:
                img_np, sample_id = self.data[idx]
                # preprocess image
                img = self.gamma_correction(img_np, gamma=0.2)
                img = img.astype(np.uint8)
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                img = cv2.GaussianBlur(img, (5,5), 0)

                aug = self.transform(image=img)
                image_tensor = aug['image']       # (3, H, W)

                return image_tensor, sample_id

    def yolo_polygon_to_mask(self, sample_id, image):
        """
        Convert YOLO polygon (.txt) to a two-channel mask:
          channel 0 = part (inner); channel 1 = whole (outer)
        """
        h, w = image.shape[:2]
        mask_whole = np.zeros((h, w), dtype=np.uint8)
        mask_part = np.zeros((h, w), dtype=np.uint8)

        # locate label file
        label_file = os.path.join(self.annotation_dir, f"{sample_id}.txt")
        if not os.path.isfile(label_file):
            return None
        with open(label_file) as f:
            for line in f:
                toks = line.strip().split()
                if len(toks) < 3 or len(toks) % 2 == 0:
                    continue
                cls = int(toks[0])
                coords = list(map(float, toks[1:]))
                pts = np.array(
                    [[int(coords[i]*w), int(coords[i+1]*h)] for i in range(0, len(coords),2)],
                    dtype=np.int32).reshape(-1,1,2)
                if cls == 1:
                    cv2.fillPoly(mask_whole, [pts], 1)
                elif cls == 0:
                    cv2.fillPoly(mask_part, [pts], 1)
        # ensure part is inside whole
        mask_part = mask_part & mask_whole
        # stack masks
        mask = np.stack([mask_whole, mask_part], axis=-1)
        assert mask.shape == (h, w, 2), f"Incorrect mask shape: {mask.shape}"
        return mask

    def gamma_correction(self, image, gamma=1.0):

        # build lookup table [0..255] → [0..255]^(1/gamma)
        table = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")

        img_norm = cv2.LUT(image, table)
        return img_norm

    def inspect_sample(self, idx=0):
        """
        Display each channel of a sample with multi-label overlay:
          - Magenta (whole root) at alpha=0.5
          - Blue (part) at alpha=0.5
        """
        img, mask, sid = self.data[idx]
        # Visualization prep
        img_vis = self.gamma_correction(img, gamma=0.2)
        img_vis = img_vis.astype(np.uint8)
        img_vis = cv2.normalize(img_vis, None, 0, 255, cv2.NORM_MINMAX)
        img_vis = cv2.GaussianBlur(img_vis, (5,5), 0)

        h, w = mask.shape[:2]
        # Create base overlay mask (BGR)
        overlay_mask = np.zeros((h, w, 3), dtype=np.uint8)

        overlay_mask[mask[:,:,0]==1] = [255, 0, 255]  # Magenta BGR(255,0,255)
        overlay_mask[mask[:,:,1]==1] = [255, 0, 0]    # Blue BGR(255,0,0)

        alpha = 0.5
        # Plot each channel
        channels = ['DAPI', 'GFP', 'TRITC'] if img_vis.shape[2]==3 else [f"Ch{i}" for i in range(img_vis.shape[2])]
        plt.figure(figsize=(18,6))
        for i, ch in enumerate(channels):
            channel = img_vis[:,:,i]
            channel_bgr = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
            combined = cv2.addWeighted(channel_bgr, 1-alpha, overlay_mask, alpha, 0)
            combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            plt.subplot(1, len(channels), i+1)
            plt.imshow(combined_rgb)
            plt.title(f"Sample {sid} - {ch}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":
    # Example usage
    train_data_folder = r'C:\Users\Yifei\Documents\new_aere_model_training_on_tamera\train'        # Contains subfolders: image, annotation.
    channels = ['DAPI', 'FITC', 'TRITC']
    train_dataset = BinarySegDataset(train_data_folder, channels,transform=get_train_transforms())
    train_dataset.inspect_sample(idx=3)