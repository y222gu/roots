import numpy as np
import torch
import cv2
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from transformers import SamModel
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.optim import Adam
import monai
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize


class ImageMaskDatasetWithPoints(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Get image and mask paths
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.image_files[idx])  # Assuming masks share filenames
        
        # Load image and mask
        image = cv2.imread(img_path)[..., ::-1]  # Convert BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
        
        if image is None or mask is None:
            raise FileNotFoundError(f"Image or mask not found: {img_path} or {mask_path}")

        # Resize to 1024x1024
        image = cv2.resize(image, (1024, 1024))
        mask = cv2.resize(mask, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        # Find unique labels in the mask and sample points
        points = []
        binary_mask = np.zeros_like(mask, dtype=np.uint8)
        for label in np.unique(mask):
            if label == 0:  # Skip background
                continue
            
            label_mask = (mask == label).astype(np.uint8)
            binary_mask = np.maximum(binary_mask, label_mask)
            
            # Erode the mask to avoid sampling boundary points
            eroded_mask = cv2.erode(label_mask, np.ones((5, 5), np.uint8), iterations=1)
            coords = np.argwhere(eroded_mask > 0)
            
            if len(coords) > 0:
                point = coords[np.random.randint(len(coords))]
                points.append([point[1], point[0]])  # Append (x, y)

        points = np.array(points) if points else np.zeros((1, 2))

        # Apply transformations if provided
        if self.transform:
            augmented = self.transform(image=image, mask=binary_mask)
            image = augmented['image']
            binary_mask = augmented['mask']

        # Convert to tensors
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0  # Normalize to [0, 1]
        binary_mask = torch.tensor(binary_mask, dtype=torch.long)
        points = torch.tensor(points, dtype=torch.float32)

        return image, binary_mask, points

if __name__ == "__main__":
    # Load data
    image_dir = os.path.join(os.getcwd(), "aerenchyma_segmentation","data","train","images") # Path to dataset (LabPics 1)
    mask_dir= os.path.join(os.getcwd(), "aerenchyma_segmentation","data","train", "masks") # Path to dataset (LabPics 1)
    dataset = ImageMaskDatasetWithPoints(image_dir, mask_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=2, shuffle=False)

    model = SamModel.from_pretrained("facebook/sam-vit-base")
    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
        if name.startswith("mask_decoder"):
            param.requires_grad_(True)

    #Training loop
    # Initialize the optimizer and the loss function
    optimizer = Adam(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    #Try DiceFocalLoss, FocalLoss, DiceCELoss
    seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    num_epochs = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.train()
    for epoch in range(num_epochs):
        epoch_losses = []
        for image, binary_mask, points in tqdm(train_dataloader):
            # forward pass
            outputs = model(pixel_values=image.to(device),
                            input_point = points.to(device),
                            multimask_output=False)

            # compute loss
            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = binary_mask.float().to(device)
            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            # backward pass (compute gradients of parameters w.r.t. loss)
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
            epoch_losses.append(loss.item())

        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

    # Save the model
    torch.save(model.state_dict(), "sam_model.pth")
