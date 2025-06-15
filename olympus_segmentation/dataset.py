import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# --------------------- Dataset Class ---------------------
class MultiChannelSegDataset(Dataset):
    def __init__(self, data_dir, channels, transform=None):
        """
        Args:
            image_dir (str): Path to the main data folder that contains three subfolders: DAPI, GFP, TRITC.
            annotation_dir (str): Path to the folder with annotation files.
            transform: Optional albumentations transform to be applied on the images and masks.
        """
        self.image_dir = os.path.join(data_dir, 'image') # Assuming images are in the same folder as annotations
        self.annotation_dir = os.path.join(data_dir, 'annotation')
        self.transform = transform
        self.channels = channels

        # Get the all subfolder names in the image directory as sample id.
        self.sample_ids = [f for f in os.listdir(self.image_dir ) if os.path.isdir(os.path.join(self.image_dir , f))]

        # loaf all the image stacks
        self.data = []
        for sample_id in self.sample_ids:
            image_stack = self.load_an_image_stack(sample_id)
            annotation = self.yolo_polygon_to_mask(sample_id, image_stack)
            if image_stack is None or annotation is None:
                print(f"Skipping sample {sample_id} due to missing data.")
                continue

            self.data.append((image_stack, annotation, sample_id))
        print(f"Found {len(self.data)} samples")


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
        return len(self.sample_ids)

    def __getitem__(self, idx):
        data = self.data[idx]
        image, mask, sample_id = data

        image = self.gamma_correction(image, gamma=0.2)
        image = image.astype(np.uint8)  # Ensure mask is in uint8 format
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply optional transforms (data augmentation, normalization, etc.)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask, sample_id

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
            if fn.endswith('.txt') and sample_id in fn:
                label_path = os.path.join(self.annotation_dir, fn)
                break
        if label_path is None:
            return None

        outer_polys = []
        inner_polys = []

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

                if cls == 1:
                    outer_polys.append(poly)
                elif cls == 0:
                    inner_polys.append(poly)
                # else: ignore any other classes

        # Fill the outer region first
        if outer_polys:
            cv2.fillPoly(mask, outer_polys, color=1)

        # Then “erase” the inner region (make it background again)
        if inner_polys:
            cv2.fillPoly(mask, inner_polys, color=0)

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
        overlay_mask[mask == 0] = [0, 0, 255]   # red for outer ring
        overlay_mask[mask == 1] = [255, 0, 0]# blue for inner ring
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


class MultiChannelSegDataset_no_annotation(Dataset):
    def __init__(self, data_dir, channels, transform=None):
        """
        Args:
            image_dir (str): Path to the main data folder that contains three subfolders: DAPI, GFP, TRITC.
            annotation_dir (str): Path to the folder with annotation files.
            transform: Optional albumentations transform to be applied on the images and masks.
        """
        self.image_dir = os.path.join(data_dir, 'image') # Assuming images are in the same folder as annotations
        self.transform = transform
        self.channels = channels

        # Get the all subfolder names in the image directory as sample id.
        self.sample_ids = [f for f in os.listdir(self.image_dir ) if os.path.isdir(os.path.join(self.image_dir , f))]

        # loaf all the image stacks
        self.data = []
        for sample_id in self.sample_ids:
            image_stack = self.load_an_image_stack(sample_id)

            self.data.append((image_stack, sample_id))
        print(f"Found {len(self.data)} samples")


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
        return len(self.sample_ids)

    def __getitem__(self, idx):
        image, sample_id = self.data[idx]

        image = self.gamma_correction(image, gamma=0.2)
        image = image.astype(np.uint8)  # Ensure mask is in uint8 format
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Apply optional transforms (data augmentation, normalization, etc.)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, sample_id

    def gamma_correction(self, image, gamma=1.0):

        # build lookup table [0..255] → [0..255]^(1/gamma)
        table = np.array([
            ((i / 255.0) ** gamma) * 255
            for i in np.arange(256)
        ]).astype("uint8")

        img_norm = cv2.LUT(image, table)
        return img_norm