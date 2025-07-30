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
class MultiChannelSegDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the main data folder that contains three subfolders: DAPI, GFP, TRITC.
            annotation_dir (str): Path to the folder with annotation files.
            transform: Optional albumentations transform to be applied on the images and masks.
        """
        self.image_dir = os.path.join(data_dir, 'image') # Assuming images are in the same folder as annotations
        self.annotation_dir = os.path.join(data_dir, 'annotation')
        self.transform = transform

        # Get the sample identifiers from the DAPI folder.
        dapi_dir = os.path.join(self.image_dir, 'DAPI')
        # List all .tif files in the DAPI folder and extract the sample identifier by removing '_DAPI' and the extension.
        files = [f for f in os.listdir(dapi_dir) if f.endswith('.tif')]
        self.sample_ids = [os.path.splitext(f)[0].replace('_DAPI', '') for f in files]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        channels = ['DAPI', 'GFP', 'TRITC']
        imgs = []
        # Load the corresponding image from each channel's subfolder.
        for ch in channels:
            img_path = os.path.join(self.image_dir, ch, f"{sample_id}_{ch}.tif")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            imgs.append(img)
        # Stack the images along the channel axis to get a multi-channel image.
        image = np.stack(imgs, axis=-1)

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
        Convert YOLO polygon annotations to a segmentation mask.
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
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Label file not found: {label_path}")

        h, w = image_shape[:2]
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
        Load and display a sample and its segmentation mask overlay on each individual channel.
        For each channel (DAPI, GFP, TRITC), this method displays the grayscale image with the segmentation
        overlay applied (red for outer ring, blue for inner ring).
        """
        sample_id = self.sample_ids[idx]
        channels = ['DAPI', 'GFP', 'TRITC']
        imgs = []
        for ch in channels:
            img_path = os.path.join(self.image_dir, ch, f"{sample_id}_{ch}.tif")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            imgs.append(img)
        image = np.stack(imgs, axis=-1)
        label_path = os.path.join(self.annotation_dir, f"{sample_id}_DAPI.txt")
        mask = self.yolo_polygon_to_mask(label_path, image)

        overlay_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        overlay_mask[mask == 1] = [255, 0, 0]  # red for outer ring
        overlay_mask[mask == 2] = [0, 0, 255]  # blue for inner ring
        alpha = 0.5

        plt.figure(figsize=(18, 6))
        for i, ch in enumerate(channels):
            channel_img = imgs[i]
            channel_img_color = cv2.cvtColor(channel_img, cv2.COLOR_GRAY2BGR)
            overlay = cv2.addWeighted(channel_img_color, 1 - alpha, overlay_mask, alpha, 0)
            plt.subplot(1, 3, i + 1)
            plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
            plt.title(f"{sample_id} - {ch}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()


# --------------------- Transforms ---------------------
def get_train_transforms():
    return A.Compose([
        A.Resize(1024, 1024),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(),
        ToTensorV2(),
    ])

# --------------------- Training Function with Early Stopping ---------------------
def train_model(train_loader, val_loader, output_dir, epochs=50, lr=1e-3, patience=6):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    print(f"Training size: {train_size}, Validation size: {val_size}")

    # Create U-Net model for 3-class segmentation.
    model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=3, activation=None)
    model.to(device)
    
    loss_fn = smp.losses.DiceLoss(mode='multiclass')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)    # shape: (B, 3, H, W)
            masks = masks.to(device).long() # shape: (B, H, W)
            optimizer.zero_grad()
            outputs = model(images)         # shape: (B, 3, H, W)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= train_size

        # Validation step.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).long()
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_loss /= val_size
        
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Check for improvement in validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model.
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print("Model saved.")
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")
    # Load the best model before returning.
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
    return model

# --------------------- Plot Predictions ---------------------
def plot_all_predictions(model, dataset, output_folder):
    """
    For each sample in the validation dataset, this function:
      - Loads the original images (from each channel) and resizes them to (1024,1024).
      - Computes the true segmentation mask from annotations.
      - Obtains the predicted mask from the model.
      - Creates comparison plots for each channel, with the left column showing the true overlay and 
        the right column showing the predicted overlay.
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
        # Load original images from file and resize to (1024,1024)
        # ------------------------------
        imgs = []
        for ch in channels:
            img_path = os.path.join(dataset.image_dir, ch, f"{sample_id}_{ch}.tif")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            # Resize to match the new transform size.
            img = cv2.resize(img, (1024, 1024))
            imgs.append(img)
        # Stack channels to get a multi-channel image.
        orig_image = np.stack(imgs, axis=-1)
        
        # ------------------------------
        # Compute the true mask from annotations.
        # ------------------------------
        label_path = os.path.join(dataset.annotation_dir, f"{sample_id}_DAPI.txt")
        true_mask = dataset.yolo_polygon_to_mask(label_path, orig_image)
        
        # ------------------------------
        # Get the predicted mask from the model.
        # Use the transformed version from the dataset.
        # ------------------------------
        image_trans, _ = dataset[idx]  # using transformed image (and mask)
        image_tensor = torch.unsqueeze(image_trans, 0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # ------------------------------
        # Create colored overlay masks.
        # ------------------------------
        def create_overlay(mask):
            overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            overlay[mask == 1] = [255, 0, 0]  # red for outer ring
            overlay[mask == 2] = [0, 0, 255]  # blue for inner ring
            return overlay
        
        true_overlay = create_overlay(true_mask)
        pred_overlay = create_overlay(pred_mask)
        alpha = 0.5  # blending factor
        
        # ------------------------------
        # Create the plot.
        # ------------------------------
        fig, axes = plt.subplots(nrows=len(channels), ncols=2, figsize=(10, 4 * len(channels)))
        fig.suptitle(f"Sample: {sample_id}", fontsize=16)
        
        for i, ch in enumerate(channels):
            # Convert channel grayscale image to BGR.
            channel_img = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2BGR)
            # Blend with true and predicted overlays.
            true_blend = cv2.addWeighted(channel_img, 1 - alpha, true_overlay, alpha, 0)
            pred_blend = cv2.addWeighted(channel_img, 1 - alpha, pred_overlay, alpha, 0)
            
            if len(channels) == 1:
                ax_true, ax_pred = axes[0], axes[1]
            else:
                ax_true, ax_pred = axes[i, 0], axes[i, 1]
            
            ax_true.imshow(cv2.cvtColor(true_blend, cv2.COLOR_BGR2RGB))
            ax_true.set_title(f"{ch} - True Mask")
            ax_true.axis("off")
            
            ax_pred.imshow(cv2.cvtColor(pred_blend, cv2.COLOR_BGR2RGB))
            ax_pred.set_title(f"{ch} - Predicted Mask")
            ax_pred.axis("off")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        save_path = os.path.join(output_folder, f"{sample_id}_comparison.png")
        plt.savefig(save_path)
        plt.close(fig)
        print(f"Saved prediction comparison for sample {sample_id} to {save_path}")

def plot_all_predictions_separately(model, dataset, output_folder):
    """
    For each sample in the validation dataset, this function:
      - Loads the original images (from each channel) and resizes them to (1024,1024).
      - Computes the true segmentation mask from annotations.
      - Obtains the predicted mask from the model.
      - Creates comparison plots for each channel (each saved as an individual image file),
        with the left plot showing the true overlay and the right plot showing the predicted overlay.
      - The plots use a black background (no white background).
      - Saves each comparison plot in the specified output folder.
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
        # Load original images from file and resize to (1024,1024)
        # ------------------------------
        imgs = []
        for ch in channels:
            img_path = os.path.join(dataset.image_dir, ch, f"{sample_id}_{ch}.tif")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            # Resize to match the new transform size.
            img = cv2.resize(img, (1024, 1024))
            imgs.append(img)
        # Stack channels to get a multi-channel image.
        orig_image = np.stack(imgs, axis=-1)
        
        # ------------------------------
        # Compute the true mask from annotations.
        # ------------------------------
        label_path = os.path.join(dataset.annotation_dir, f"{sample_id}_DAPI.txt")
        true_mask = dataset.yolo_polygon_to_mask(label_path, orig_image)
        
        # ------------------------------
        # Get the predicted mask from the model.
        # Use the transformed version from the dataset.
        # ------------------------------
        image_trans, _ = dataset[idx]  # using transformed image (and mask)
        image_tensor = torch.unsqueeze(image_trans, 0).to(device)
        with torch.no_grad():
            output = model(image_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # ------------------------------
        # Create colored overlay masks.
        # ------------------------------
        def create_overlay(mask):
            overlay = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            overlay[mask == 1] = [255, 0, 0]  # red for outer ring
            overlay[mask == 2] = [0, 0, 255]  # blue for inner ring
            return overlay
        
        true_overlay = create_overlay(true_mask)
        pred_overlay = create_overlay(pred_mask)
        alpha = 0.5  # blending factor
        
        # For each channel, create an individual plot.
        for i, ch in enumerate(channels):
            # Convert channel grayscale image to BGR.
            channel_img = cv2.cvtColor(imgs[i], cv2.COLOR_GRAY2BGR)
            # Blend with true and predicted overlays.
            true_blend = cv2.addWeighted(channel_img, 1 - alpha, true_overlay, alpha, 0)
            pred_blend = cv2.addWeighted(channel_img, 1 - alpha, pred_overlay, alpha, 0)
            
            # Create a new figure with a black background.
            fig, axes = plt.subplots(ncols=2, figsize=(10, 4), facecolor='black')
            # Set axes background to black.
            axes[0].set_facecolor('black')
            axes[1].set_facecolor('black')
            
            # Display the true blend.
            axes[0].imshow(cv2.cvtColor(true_blend, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"{ch} - True Mask", color='white')
            axes[0].axis("off")
            
            # Display the predicted blend.
            axes[1].imshow(cv2.cvtColor(pred_blend, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"{ch} - Predicted Mask", color='white')
            axes[1].axis("off")
            
            # Add an overall title to the figure.
            fig.suptitle(f"Sample: {sample_id} - {ch}", color='white', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the figure with the black background.
            save_path = os.path.join(output_folder, f"{sample_id}_{ch}_comparison.png")
            plt.savefig(save_path, facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"Saved prediction comparison for sample {sample_id}, channel {ch} to {save_path}")


# --------------------- Main ---------------------
if __name__ == '__main__':
    # Set your directories.
    train_data_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\train'        # Contains subfolders: image, annotation.
    val_data_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\original\Folder1_processed\val'         # Contains subfolders: image, annotation.
    output_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\new_val'          # Directory to save the best model.
    model_path = os.path.join(output_folder, 'best_model.pth')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # image_dir = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\Folder1_processed\all_cropped'  # Path to the main data folder that contains three subfolders: DAPI, GFP, TRITC.
    # annotation_dir = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\annotation_folder1_cropped'
    # # Optionally inspect a sample before training.
    # dataset_inspect = MultiChannelSegDataset(image_dir=image_dir, annotation_dir=annotation_dir)
    # dataset_inspect.inspect_sample(idx=2)
    
    # initialize the dataset with transforms.
    train_dataset = MultiChannelSegDataset(train_data_folder, transform=get_train_transforms())
    val_dataset = MultiChannelSegDataset(val_data_folder, transform=get_val_transforms())
    # val_dataset.inspect_sample(idx=0)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)
    
    # Train the model with early stopping.
    # best_model = train_model(train_loader, val_loader, output_folder, epochs=50, lr=1e-3, patience=6)

    # # load the best model for inference.
    best_model = smp.Unet(encoder_name='resnet34', in_channels=3, classes=3, activation=None)
    best_model.load_state_dict(torch.load(model_path))
    
    # Plot predictions on a few samples from the test (validation) dataset.
    # plot_all_predictions(best_model, val_dataset, output_folder)
    plot_all_predictions_separately(best_model, val_dataset, output_folder)
    print("All predictions plotted and saved.")
