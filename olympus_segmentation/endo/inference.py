import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json  # Import here if not already imported at the top of the file
# Import your custom modules
from utils import MultiChannelSegDataset, create_model
from transforms_for_hyper_training import get_val_transforms, get_org_transforms
from loss_functions import DiceBCELoss
from pathlib import Path
import csv


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model info from checkpoint
    architecture = checkpoint.get('architecture', 'Unet')
    encoder = checkpoint.get('encoder', 'resnet34')
    n_channels = 3  # Adjust if different
    n_classes = 2   # Adjust if different
    
    # Create and load model
    model = create_model(encoder, architecture, n_channels, n_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded: {architecture} with {encoder}")
    print(f"Training Dice: {checkpoint.get('best_dice', 'Unknown'):.4f}")
    
    return model


def compute_metrics(preds, targets, threshold=0.5):
    """Compute segmentation metrics"""
    preds_binary = (preds > threshold).float()
    
    smooth = 1e-6
    intersection = (preds_binary * targets).sum(dim=(2, 3))
    union = preds_binary.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    
    # Dice and IoU
    dice = (2 * intersection + smooth) / (union + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    
    # Precision, Recall
    tp = intersection
    fp = (preds_binary * (1 - targets)).sum(dim=(2, 3))
    fn = ((1 - preds_binary) * targets).sum(dim=(2, 3))
    
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    
    # Accuracy
    accuracy = (preds_binary == targets).float().mean(dim=(1, 2, 3))
    
    return {
        'dice': dice.mean(dim=1).cpu().numpy(),
        'iou': iou.mean(dim=1).cpu().numpy(),
        'precision': precision.mean(dim=1).cpu().numpy(),
        'recall': recall.mean(dim=1).cpu().numpy(),
        'f1': f1.mean(dim=1).cpu().numpy(),
        'accuracy': accuracy.cpu().numpy()
    }


def save_predictions(imgs, probs, masks, sample_metadata, output_dir):
    """Save prediction overlay images"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy
    imgs_np = imgs.cpu().numpy()
    probs_np = probs.cpu().numpy()
    masks_np = masks.cpu().numpy()
    
    for i in range(imgs.size(0)):
        sample_name = f"{sample_metadata['microscope'][i]}_{sample_metadata['species'][i]}_{sample_metadata['genotype'][i]}_{sample_metadata['sample_id'][i]}"
        
        # Create figure with input, prediction, ground truth, and overlay
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes = axes.flatten()

        # # Input image (first channel)
        # rgb_img = np.transpose(imgs_np[i][:3], (1, 2, 0))
        # axes[0].imshow(rgb_img)
        # axes[0].set_title(f'Input RGB\n{sample_id}')
        # axes[0].axis('off')

        # Ground truth overlay
        gt_overlay = np.zeros((*masks_np.shape[2:], 3), dtype=np.uint8)
        if masks_np.shape[1] >= 2:
            gt_overlay[masks_np[i, 0] == 1] = [0, 0, 255]  # Blue for class 0
            gt_overlay[masks_np[i, 1] == 1] = [255, 0, 255]  # Magenta for class 1
        axes[0].imshow(gt_overlay)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        # Prediction overlay
        pred_binary = (probs_np[i] > 0.5).astype(np.uint8)
        pred_overlay = np.zeros((*pred_binary.shape[1:], 3), dtype=np.uint8)
        if pred_binary.shape[0] >= 2:
            pred_overlay[pred_binary[0] == 1] = [0, 0, 255]  # Blue for class 0
            pred_overlay[pred_binary[1] == 1] = [255, 0, 255]  # Magenta for class 1
        axes[1].imshow(pred_overlay)
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(output_dir / f'{sample_name}_masks.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(16, 8))
        axes = axes.flatten()
        # Combined overlay on input
        axes[0].imshow(imgs_np[i, 0], cmap='gray', alpha=0.7)
        axes[0].imshow(pred_overlay, alpha=0.5)
        axes[0].set_title('Prediction on DAPI Channel')
        axes[0].axis('off')

        axes[1].imshow(imgs_np[i, 1], cmap='gray', alpha=0.7)
        axes[1].imshow(pred_overlay, alpha=0.5)
        axes[1].set_title('Prediction on FITC Channel')
        axes[1].axis('off')

        axes[2].imshow(imgs_np[i, 2], cmap='gray', alpha=0.7)
        axes[2].imshow(pred_overlay, alpha=0.5)
        axes[2].set_title('Prediction on TRITC Channel')
        axes[2].axis('off')

        # Remove gap between subplots and extra margins
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(output_dir / f'{sample_name}_overlay.png', dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()


def run_inference(model, test_loader, device, channels, save_images=False, output_dir=None):
    """Run inference and compute metrics, saving per-image metrics to a JSON file if output_dir is provided.
    Also saves average intensity under predicted mask for each channel of each image to a CSV.
    """
    loss_fn = DiceBCELoss()
    all_metrics = {'dice': [], 'iou': [], 'precision': [], 'recall': [], 'f1': [], 'accuracy': []}
    total_loss = 0.0
    num_samples = 0

    # To store per-image metrics
    all_sample_metrics = []
    # To store per-image, per-channel average intensity under predicted mask
    avg_intensity_rows = []

    print("Running inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if len(batch) == 4:
                imgs, masks, sample_metadata, imgs_org = batch
                imgs = imgs.to(device)
                masks = masks.to(device)

                # Forward pass
                logits = model(imgs)
                probs = torch.sigmoid(logits)

                # # Save overlay images if requested
                if save_images and output_dir:
                    save_predictions(imgs, probs, masks, sample_metadata, output_dir)

                # Compute loss
                loss = loss_fn(logits, masks)
                total_loss += loss.item() * imgs.size(0)
                num_samples += imgs.size(0)

                # Compute metrics for this batch
                batch_metrics = compute_metrics(probs, masks)

                # Accumulate metrics across batches
                for key in all_metrics:
                    all_metrics[key].extend(batch_metrics[key])

                # Save per-image metrics for this batch
                batch_size = imgs.size(0)
                for i in range(batch_size):
                    # Use provided sample_ids if available, else generate one
                    sample_name = f"{sample_metadata['microscope'][i]}_{sample_metadata['species'][i]}_{sample_metadata['genotype'][i]}_{sample_metadata['sample_id'][i]}"
                    image_metrics = {
                        'sample_name': sample_name,
                        'dice': batch_metrics['dice'][i],
                        'iou': batch_metrics['iou'][i],
                        'precision': batch_metrics['precision'][i],
                        'recall': batch_metrics['recall'][i],
                        'f1': batch_metrics['f1'][i],
                        'accuracy': batch_metrics['accuracy'][i]
                    }
                    all_sample_metrics.append(image_metrics)

            elif len(batch) == 3:
                imgs, sample_metadata, imgs_org = batch
                imgs = imgs.to(device)
                # Create dummy masks if not available
                masks = torch.zeros((imgs.size(0), 2, imgs.size(2), imgs.size(3)), device=device)

                # Forward pass
                logits = model(imgs)
                probs = torch.sigmoid(logits)

                # # Save overlay images if requested
                if save_images and output_dir:
                    save_predictions(imgs, probs, masks, sample_metadata, output_dir)

            # Compute average intensity under predicted mask for each class separately
            # imgs: (B, C, H, W), probs: (B, n_classes, H, W)
            # For each class (assume 2 classes: 0 and 1)
            batch_size = imgs.size(0)
            for i in range(batch_size):
                for cls in range(probs.shape[1]):
                    pred_mask = (probs[i, cls] > 0.5).cpu().numpy()  # shape (H, W)
                    for ch in range(imgs_org.shape[1]):
                        img_ch = imgs_org[i, ch].cpu().numpy()
                        if pred_mask.sum() > 0:
                            avg_intensity = float(img_ch[pred_mask > 0].mean())
                        else:
                            avg_intensity = float('nan')
                        class_name = 'vascular' if cls == 0 else 'endo'
                        channel_name = channels[ch] if ch < len(channels) else f"channel_{ch}"
                        sample_name = f"{sample_metadata['microscope'][i]}_{sample_metadata['species'][i]}_{sample_metadata['genotype'][i]}_{sample_metadata['sample_id'][i]}"
                        avg_intensity_rows.append({
                            'sample_name': sample_name,
                            'class': class_name,
                            'channel': channel_name,
                            'avg_intensity_under_pred_mask': avg_intensity
                        })

    # Calculate final averages
    if len(all_sample_metrics) > 0:
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        final_metrics = {key: np.mean(values) for key, values in all_metrics.items()}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_json_path = output_dir / "image_metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(all_sample_metrics, f, indent=4, default=lambda o: float(o) if isinstance(o, np.float32) else o)
        # Print results
        print_results(avg_loss, final_metrics)

    # Save average intensity under predicted mask to CSV
    output_dir = Path(output_dir)
    intensity_csv_path = output_dir / "avg_intensity_under_pred_mask.csv"
    with open(intensity_csv_path, "w", newline="") as csvfile:
        fieldnames = ['sample_name', 'class', 'channel', 'avg_intensity_under_pred_mask']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in avg_intensity_rows:
            writer.writerow(row)
    print(f"Average intensity under predicted mask saved to: {intensity_csv_path}")


def print_results(avg_loss, metrics):
    """Print results in a clean format"""
    print("\n" + "="*50)
    print("INFERENCE RESULTS")
    print("="*50)
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Dice Score:   {metrics['dice']:.4f}")
    print(f"IoU Score:    {metrics['iou']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print(f"Accuracy:     {metrics['accuracy']:.4f}")
    print("="*50)


def main():
    """Main inference function"""
    
    # MODIFY THESE PATHS
    checkpoint_path = r"C:\Users\yifei\Documents\data_for_publication\train_on_sorghum_and_rice\results\models\Unet_resnet34_2025_08_05_trained_without_channel_dropout\continued_training\continued_best_model_loss_0.1026_dice_0.9189_epoch_247.pth"
    test_data_dir = r"C:\Users\yifei\Documents\data_for_publication\train_on_sorghum_and_rice\test"
    output_dir = r"C:\Users\yifei\Documents\data_for_publication\train_on_sorghum_and_rice\results\inference_overlays"

    # Configuration
    channels = ['DAPI', 'FITC', 'TRITC']
    batch_size = 4
    save_overlay_images = True  # Set to False if you don't want to save images
    manual_annotation = True  # Set to True if test data has manual annotations for metrics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(checkpoint_path, device)
    
    # Create test dataset
    test_dataset = MultiChannelSegDataset(
        test_data_dir,
        channels,
        transform_for_pred=get_val_transforms(),
        transform_for_org=get_org_transforms(),
        manual_annotation=manual_annotation
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )
    
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Run inference
    run_inference(
        model, test_loader, device, 
        channels=channels,
        save_images=save_overlay_images, 
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()
