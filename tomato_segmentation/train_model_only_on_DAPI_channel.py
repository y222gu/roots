import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

# --------------------- Dataset Class ---------------------
class SingleChannelSegDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the main data folder that contains three subfolders: DAPI, GFP, TRITC.
            annotation_dir (str): Path to the folder with annotation files.
            transform: Optional albumentations transform to be applied on the images and masks.
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform

        # Get the sample identifiers from the DAPI folder.
        dapi_dir = os.path.join(image_dir, 'DAPI')
        # List all .tif files in the DAPI folder and extract the sample identifier by removing '_DAPI' and the extension.
        files = [f for f in os.listdir(dapi_dir) if f.endswith('.tif')]
        self.sample_ids = [os.path.splitext(f)[0].replace('_DAPI', '') for f in files]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        # Load only the DAPI channel image.
        img_path = os.path.join(self.image_dir, 'DAPI', f"{sample_id}_DAPI.tif")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Build the path for the annotation file.
        label_path = os.path.join(self.annotation_dir, f"{sample_id}_DAPI.txt")
        mask = self.yolo_polygon_to_mask(label_path, image)

        # Apply optional transforms (data augmentation, normalization, etc.)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

    def yolo_polygon_to_mask(self, label_path, image):
        """
        Convert YOLO polygon annotations to a segmentation mask for a single-channel image.
        Each line in the label file is expected to be:
            <class> <x1> <y1> <x2> <y2> ... <xn> <yn>
        where coordinates are normalized (0 to 1).

        For your donut structure:
        - Class 1 annotations represent the outer ring of the donut.
        - Class 2 annotations represent the inner ring of the donut.

        The output mask is of shape (height, width) with:
            0 = background
            1 = outer ring (class 1)
            2 = inner ring (class 2)

        The function fills class 1 polygons first and then overlays class 2 polygons.
        """
        image_shape = image.shape
        mask = np.zeros(image_shape, dtype=np.uint8)  # Single-channel mask
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        h, w = image_shape
        # Store polygons separately for class 1 and class 2
        polygons_class1 = []
        polygons_class2 = []
        with open(label_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) < 3 or len(tokens) % 2 == 0:
                    continue  # skip invalid lines
                cls = int(tokens[0])
                coords = list(map(float, tokens[1:]))
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    points.append([x, y])
                polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                if cls == 1:
                    polygons_class1.append(polygon)
                elif cls == 2:
                    polygons_class2.append(polygon)
        if polygons_class1:
            cv2.fillPoly(mask, polygons_class1, color=1)
        if polygons_class2:
            cv2.fillPoly(mask, polygons_class2, color=2)
        return mask

    def inspect_sample(self, idx=0):
        """
        Load and display a single-channel image and its segmentation mask overlay.
        This method displays the grayscale image with the segmentation overlay applied
        (red for outer ring, blue for inner ring).
        """
        sample_id = self.sample_ids[idx]
        img_path = os.path.join(self.image_dir, 'DAPI', f"{sample_id}_DAPI.tif")
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        label_path = os.path.join(self.annotation_dir, f"{sample_id}_DAPI.txt")
        mask = self.yolo_polygon_to_mask(label_path, image)

        # Create an overlay mask
        overlay_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        overlay_mask[mask == 1] = [255, 0, 0]  # red for outer ring
        overlay_mask[mask == 2] = [0, 0, 255]  # blue for inner ring
        alpha = 0.5

        # Convert grayscale image to BGR for visualization
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(image_color, 1 - alpha, overlay_mask, alpha, 0)

        # Plot the image with overlay
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"{sample_id} - DAPI with Overlay")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


# --------------------- Transforms ---------------------
def get_train_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(),
        ToTensorV2(),
    ])

# --------------------- Training Function with Early Stopping ---------------------
def train_model(train_dataloader, val_dataloader, output_dir, epochs=50, lr=1e-3, patience=6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_size = len(train_dataloader.dataset)
    val_size = len(val_dataloader.dataset)
    print(f"Training size: {train_size}, Validation size: {val_size}")

    # Create U-Net model for segmentation with single-channel input.
    model = smp.Unet(encoder_name='resnet34', in_channels=1, classes=3, activation=None)
    model.to(device)
    
    loss_fn = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Optional: use a learning rate scheduler to reduce LR when the validation loss plateaus.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_dataloader:
            images = images.to(device)    # images shape: (B, 1, H, W)
            masks = masks.to(device).long() # masks shape: (B, H, W)
            optimizer.zero_grad()
            outputs = model(images)         # outputs shape: (B, 3, H, W)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= train_size

        # Validation step.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_dataloader:
                images = images.to(device)
                masks = masks.to(device).long()
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= val_size
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Update scheduler based on validation loss.
        scheduler.step(val_loss)

        # Check for improvement in validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model.
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print("Model saved.")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    # Load the best model before returning.
    model.load_state_dict(torch.load(best_model_path))
    return model, val_dataloader

# --------------------- Plot Predictions ---------------------

def plot_all_predictions(model, dataset, output_folder):
    """
    For each sample in the validation dataset, this function:
      - Loads the original images from each channel and resizes them to (256,256).
      - Computes the true segmentation mask from annotations.
      - Obtains the predicted mask using only the DAPI channel input (since the model is trained on single-channel images).
      - For each channel (DAPI, GFP, TRITC), creates a two-column plot:
            Left column: original channel image with true mask overlay.
            Right column: original channel image with predicted mask overlay.
      - Saves each comparison plot as an image file in the specified output folder.
    """
    model.eval()
    device = next(model.parameters()).device
    channels = ['DAPI', 'GFP', 'TRITC']
    
    # Ensure the output folder exists.
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate over all samples in the validation dataset.
    for idx in range(len(dataset)):
        # Get sample identifier.
        sample_id = dataset.sample_ids[idx] if hasattr(dataset, 'sample_ids') else dataset.dataset.sample_ids[idx]
        
        # ------------------------------
        # Load original images from file and resize to (256,256)
        # ------------------------------
        imgs = []
        for ch in channels:
            img_path = os.path.join(dataset.image_dir, ch, f"{sample_id}_{ch}.tif")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            # Resize to match the training transform size.
            img = cv2.resize(img, (256, 256))
            imgs.append(img)
        # Stack channels (for computing the true mask).
        orig_image = np.stack(imgs, axis=-1)
        
        # ------------------------------
        # Compute the true mask from annotations.
        # ------------------------------
        label_path = os.path.join(dataset.annotation_dir, f"{sample_id}_DAPI.txt")
        true_mask = dataset.yolo_polygon_to_mask(label_path, orig_image)
        
        # ------------------------------
        # Get the predicted mask from the model using only the DAPI channel.
        # ------------------------------
        # Assume that when training on single-channel images, dataset[idx] returns an image of shape (1, H, W).
        image_trans, _ = dataset[idx]  # image_trans is the transformed DAPI channel image.
        image_tensor = torch.unsqueeze(image_trans, 0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        # pred_mask is expected to be (256,256) with labels {0,1,2}.
        
        # ------------------------------
        # Create colored overlay masks.
        # ------------------------------
        def create_overlay(mask):
            overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            overlay[mask == 1] = [255, 0, 0]    # red for outer ring (class 1)
            overlay[mask == 2] = [0, 0, 255]    # blue for inner ring (class 2)
            return overlay
        
        true_overlay = create_overlay(true_mask)
        pred_overlay = create_overlay(pred_mask)
        alpha = 0.5  # blending factor
        
        # ------------------------------
        # Create the plot: for each channel, show two columns (True overlay | Predicted overlay)
        # ------------------------------
        fig, axes = plt.subplots(nrows=len(channels), ncols=2, figsize=(10, 4 * len(channels)))
        fig.suptitle(f"Sample: {sample_id}", fontsize=16)
        
        for i, ch in enumerate(channels):
            # Convert channel grayscale image to BGR.
            channel_img = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2BGR)
            # Blend with true overlay and predicted overlay.
            true_blend = cv2.addWeighted(channel_img, 1 - alpha, true_overlay, alpha, 0)
            pred_blend = cv2.addWeighted(channel_img, 1 - alpha, pred_overlay, alpha, 0)
            
            # If there is only one row, axes might be one-dimensional.
            if len(channels) == 1:
                ax_true, ax_pred = axes[0], axes[1]
            else:
                ax_true, ax_pred = axes[i, 0], axes[i, 1]
            
            ax_true.imshow(cv2.cvtColor(true_blend, cv2.COLOR_BGR2RGB))
            ax_true.set_title(f"{ch} - True Mask")
            ax_true.axis("off")
            
            ax_pred.imshow(cv2.cvtColor(pred_blend, cv2.COLOR_BGR2RGB))
            ax_pred.set_title(f"{ch} - Predicted Mask (from DAPI input)")
            ax_pred.axis("off")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        # Save the figure.
        save_path = os.path.join(output_folder, f"{sample_id}_comparison.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved prediction comparison for sample {sample_id} to {save_path}")


# --------------------- Main ---------------------
if __name__ == '__main__':
    # Set your directories.
    train_data_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\Folder1_processed\train'        # Contains subfolders: DAPI, GFP, TRITC.
    val_data_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\Folder1_processed\val'        # Contains subfolders: DAPI, GFP, TRITC.
    annotation_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\annotation_folder1'   # Contains annotation text files.
    output_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\training'          # Directory to save the best model.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # # Optionally inspect a sample before training.
    # dataset_inspect = MultiChannelSegDataset(image_dir=IMAGE_DIR, annotation_dir=ANNOTATION_DIR)
    # dataset_inspect.inspect_sample(idx=0)
    
    # initialize the dataset with transforms.
    train_dataset = SingleChannelSegDataset(train_data_folder, annotation_folder, transform=get_train_transforms())
    val_dataset = SingleChannelSegDataset(val_data_folder, annotation_folder, transform=get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    # Train the model with early stopping.
    best_model = train_model(train_loader, val_loader, output_folder, epochs=50, lr=1e-3, patience=6)
    
    # Plot predictions on a few samples from the test (validation) dataset.
    plot_all_predictions(best_model, val_dataset, output_folder)
    print("All predictions plotted and saved.")
