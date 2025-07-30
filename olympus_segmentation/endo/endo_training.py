import os
import torch
import segmentation_models_pytorch as smp
from torch.amp import autocast, GradScaler
from segmentation_models_pytorch.utils.metrics import IoU
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import DataLoader
from olympus_segmentation.aere.aere_dataset import BinarySegDataset
from olympus_segmentation.transforms import get_train_transforms, get_val_transforms
from  olympus_segmentation.aere.visualizing_aere_with_binary_models import visualize_all_predictions_with_manual_annotation

# --------------------- Training Function with Early Stopping ---------------------

def train_model(train_loader, val_loader, output_dir,
                epochs=100, lr=1e-3, patience=6):
    """
    Train a 2-class (binary) U-Net segmentation model with:
      - mixed precision (AMP) via torch.amp.autocast
      - combined Dice + BCEWithLogits loss
      - ReduceLROnPlateau scheduler
      - IoU metric logging
      - early stopping on val loss
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_size = len(train_loader.dataset)
    val_size   = len(val_loader.dataset)
    print(f"Train size: {train_size}, Val size: {val_size}")

    # 1-channel (binary) output, raw logits
    model = smp.Unet(
        encoder_name='resnet34',
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    # Combined Dice + BCEWithLogits
    # single Dice loss
    loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    scaler    = GradScaler()
    metric_fn = IoU(threshold=0.5)

    best_val_loss      = float('inf')
    early_stop_counter = 0

    for epoch in range(1, epochs+1):
        # ——— Training ———
        model.train()
        running_train_loss = 0.0

        for images, masks, sample_id, images_original in train_loader:
            images = images.to(device)
            masks  = masks.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            # use new autocast API
            with autocast(device_type=device.type):
                logits = model(images)
                loss   = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item() * images.size(0)

        train_loss = running_train_loss / train_size

        # ——— Validation ———
        model.eval()
        running_val_loss = 0.0
        total_iou        = 0.0

        with torch.no_grad():
            for images, masks, sample_id, images_original in val_loader:
                images = images.to(device)
                masks  = masks.to(device).unsqueeze(1).float()

                logits = model(images)
                loss   = loss_fn(logits, masks)
                running_val_loss += loss.item() * images.size(0)

                # compute IoU on probabilities
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                total_iou += metric_fn(preds, masks).item()

        val_loss = running_val_loss / val_size
        avg_iou  = total_iou / len(val_loader)

        print(
            f"Epoch [{epoch}/{epochs}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Loss:   {val_loss:.4f}  "
            f"IoU:        {avg_iou:.4f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss      = val_loss
            early_stop_counter = 0
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(),
                       os.path.join(output_dir, 'best_model.pth'))
            print("  → New best model saved.")
        else:
            early_stop_counter += 1
            print(f"  EarlyStopping: {early_stop_counter}/{patience}")
            if early_stop_counter >= patience:
                print("  → Stopping early.")
                break

    # Load best model weights before returning
    model.load_state_dict(
        torch.load(os.path.join(output_dir, 'best_model.pth'))
    )
    return model


# --------------------- Main ---------------------
if __name__ == '__main__':
    # Set your directories.
    channels = ['DAPI', 'FITC', 'TRITC']
    model_path =os.path.join(os.path.dirname(__file__), "weights", 'aere_model_for_olympus.pth')
    train_data_folder = r'C:\Users\Yifei\Documents\new_endo_model\train'        # Contains subfolders: image, annotation.
    val_data_folder = r'C:\Users\Yifei\Documents\new_endo_model\val'         # Contains subfolders: image, annotation.          # Contains subfolders: image, annotation.
    output_folder = r'C:\Users\Yifei\Documents\new_endo_model\results'          # Directory to save the best model.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
 
    # initialize the dataset with transforms.
    train_dataset = BinarySegDataset(train_data_folder, channels,transform=get_train_transforms())
    val_dataset = BinarySegDataset(val_data_folder, channels, transform=get_val_transforms())
    # train_dataset.inspect_sample(idx=3)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    # Train the model with early stopping.
    best_model = train_model(train_loader, val_loader, output_folder, epochs=100, lr=1e-3, patience=6)

    print("Training complete. Best model saved.")