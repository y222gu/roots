import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from utils import MultiChannelSegDataset, create_model
from transforms_for_hyper_training import get_val_transforms
from loss_functions import DiceBCELoss
from pathlib import Path
import json


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


def compute_metrics(preds, targets, threshold=0.99):
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


def run_inference_by_label(model, model_name, test_loader, device):
    """
    Run inference on each image, compute metrics, and group results by the specified label (e.g., "microscope").
    Optionally saves per-image and grouped metrics in JSON format.
    """

    loss_fn = DiceBCELoss()
    all_sample_metrics = []
    grouped_data = {}
    total_loss = 0.0
    num_samples = 0

    print("Running inference...")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            if len(batch) != 3:
                print("Skipping batch without required masks or meta_data")
                continue

            imgs, masks, meta_data = batch
            # Use provided sample IDs or generate defaults
            sample_ids = meta_data.get("sample_id", [f"batch_{batch_idx}_sample_{i}" for i in range(imgs.size(0))])
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            probs = torch.sigmoid(logits)
            loss = loss_fn(logits, masks)
            total_loss += loss.item() * imgs.size(0)
            num_samples += imgs.size(0)

            # Compute metrics for the current batch
            batch_metrics = compute_metrics(probs, masks)
            batch_size = imgs.size(0)

            for i in range(batch_size):
                current_id = sample_ids[i] if isinstance(sample_ids, (list, tuple)) else f"batch_{batch_idx}_sample_{i}"
                current_meta = {key: value[i] for key, value in meta_data.items()}

                image_metrics = {
                    "model_name": model_name,
                    "sample_id": current_id,
                    "dice": float(batch_metrics['dice'][i]),
                    "iou": float(batch_metrics['iou'][i]),
                    "precision": float(batch_metrics['precision'][i]),
                    "recall": float(batch_metrics['recall'][i]),
                    "f1": float(batch_metrics['f1'][i]),
                    "accuracy": float(batch_metrics['accuracy'][i]),
                    "loss": float(loss)
                }
                image_metrics.update(current_meta)
                all_sample_metrics.append(image_metrics)


    # Compute average loss across all samples
    avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

    return avg_loss, all_sample_metrics


def main():
    """Main inference function"""

    image_size_list = [50, 100, 148, 198, 232]

    # MODIFY THESE PATHS
    checkpoint_path_list = [
        r"C:\Users\yifei\Documents\data_for_publication\only_train_on_sorghum\output_train_50\models\Unet_resnet34\best_model_loss_0.7590_dice_0.7950_epoch_342.pth",
        r"C:\Users\yifei\Documents\data_for_publication\only_train_on_sorghum\output_train_100\models\Unet_resnet34\best_model_loss_0.1940_dice_0.8681_epoch_347.pth",
        r"C:\Users\yifei\Documents\data_for_publication\only_train_on_sorghum\output_train_148\models\Unet_resnet34\best_model_loss_0.1707_dice_0.8691_epoch_347.pth",
        r"C:\Users\yifei\Documents\data_for_publication\only_train_on_sorghum\output_train_198\models\Unet_resnet34\best_model_loss_0.1964_dice_0.8621_epoch_269.pth",
        r"C:\Users\yifei\Documents\data_for_publication\only_train_on_sorghum\output_train_232\models\Unet_resnet34\best_model_loss_0.1336_dice_0.8902_epoch_343.pth"
    ]
    train_data_dir = r"C:\Users\yifei\Documents\data_for_publication\only_train_on_sorghum\train_preprocessed"
    test_data_dir = r"C:\Users\yifei\Documents\data_for_publication\only_train_on_sorghum\test_preprocessed"
    output_dir = r"C:\Users\yifei\Documents\data_for_publication\only_train_on_sorghum\results"

    # Configuration
    channels = ['DAPI', 'FITC', 'TRITC']
    batch_size = 4
    save_overlay_images = False  # Set to False if you don't want to save images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    train_dataset = MultiChannelSegDataset(
        train_data_dir,
        channels,
        transform=get_val_transforms(),
        manual_annotation=True
    )

    # Create test dataset
    test_dataset = MultiChannelSegDataset(
        test_data_dir,
        channels,
        transform=get_val_transforms(),
        manual_annotation=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"Test dataset: {len(test_dataset)} samples")

    train_metrics = []
    train_avg_losses = []
    test_metrics = []
    test_avg_losses = []
    # Load model and run inference for each checkpoint
    for i, checkpoint_path in enumerate(checkpoint_path_list):
        model_name = image_size_list[i]
        model = load_model(checkpoint_path, device)

        train_avg_loss, train_grouped_metrics = run_inference_by_label(
            model, model_name, train_loader, device
        )

        # Run inference
        test_avg_loss, test_grouped_metrics = run_inference_by_label(
            model, model_name, test_loader, device
        )
        train_avg_losses.append(train_avg_loss)
        # Change all 'species' values in train_grouped_metrics to 'Olympus'

        train_metrics.append(train_grouped_metrics)
        test_metrics.append(test_grouped_metrics)
        test_avg_losses.append(test_avg_loss)

    # List of metrics to plot
    metric_names = ['dice', 'iou', 'loss']

    # Determine unique species from the train metrics (use "Unknown" if not provided)
    species_set = set()
    for group in test_metrics:
        for m in group:
            species_set.add(m.get("species", "Unknown"))
    species_list = sorted(list(species_set))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 5))
    for ax, metric in zip(axes, metric_names):
        ax.set_title(f"{metric.capitalize()} vs Training Data Size")
        ax.set_xlabel("Training Data Size")
        ax.set_ylabel(metric.capitalize())
        for idx, species in enumerate(species_list):
            train_values = []
            test_values = []
            for train_group, test_group in zip(train_metrics, test_metrics):
                train_species_metric = [m[metric] for m in train_group if m.get("species", "Unknown") == species]
                test_species_metric = [m[metric] for m in test_group if m.get("species", "Unknown") == species]
                train_values.append(np.mean(train_species_metric) if train_species_metric else np.nan)
                test_values.append(np.mean(test_species_metric) if test_species_metric else np.nan)
            if not np.all(np.isnan(train_values)):
                ax.plot(image_size_list, train_values, marker='o', label=f"Train {species}",
                        linestyle='-', color=colors[idx % len(colors)])
            if not np.all(np.isnan(test_values)):
                ax.plot(image_size_list, test_values, marker='s', label=f"Test {species}",
                        linestyle='dashed', color=colors[idx % len(colors)])
        ax.set_xticks(image_size_list)
        ax.legend()
    plt.tight_layout()
    plt.show()

    # Determine unique species from the train metrics (use "Unknown" if not provided)
    microscope_set = set()
    for group in test_metrics:
        for m in group:
            microscope_set.add(m.get("genotype", "Unknown"))
    microscope_list = sorted(list(microscope_set))
    
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for ax, metric in zip(axes, metric_names):
        ax.set_title(f"{metric.capitalize()} vs Training Data Size")
        ax.set_xlabel("Training Data Size")
        ax.set_ylabel(metric.capitalize())
        for idx, microscope in enumerate(microscope_list):
            train_values = []
            test_values = []
            for train_group, test_group in zip(train_metrics, test_metrics):
                train_microscope_metric = [m[metric] for m in train_group if m.get("genotype", "Unknown") == microscope]
                test_microscope_metric = [m[metric] for m in test_group if m.get("genotype", "Unknown") == microscope]
                train_values.append(np.mean(train_microscope_metric) if train_microscope_metric else np.nan)
                test_values.append(np.mean(test_microscope_metric) if test_microscope_metric else np.nan)
            if not np.all(np.isnan(train_values)):
                ax.plot(image_size_list, train_values, marker='o', label=f"Train {microscope}", linestyle='-', color=colors[idx % len(colors)])
            if not np.all(np.isnan(test_values)):
                ax.plot(image_size_list, test_values, marker='s', label=f"Test {microscope}", linestyle='dashed', color=colors[idx % len(colors)])
        ax.set_xticks(image_size_list)
        ax.legend()
    plt.tight_layout()
    plt.show()

        # Get unique (species, genotype) combinations from test metrics
    combo_set = set()
    for group in test_metrics:
        for m in group:
            combo = (m.get("species", "Unknown"), m.get("genotype", "Unknown"))
            combo_set.add(combo)
    combo_list = sorted(list(combo_set))

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axes = plt.subplots(1, len(metric_names), figsize=(6 * len(metric_names), 5))
    for ax, metric in zip(axes, metric_names):
        ax.set_title(f"{metric.capitalize()} vs Training Data Size")
        ax.set_xlabel("Training Data Size")
        ax.set_ylabel(metric.capitalize())
        for idx, (species, genotype) in enumerate(combo_list):
            train_values = []
            test_values = []
            for train_group, test_group in zip(train_metrics, test_metrics):
                train_combo_metric = [m[metric] for m in train_group if m.get("species", "Unknown") == species and m.get("genotype", "Unknown") == genotype]
                test_combo_metric = [m[metric] for m in test_group if m.get("species", "Unknown") == species and m.get("genotype", "Unknown") == genotype]
                train_values.append(np.mean(train_combo_metric) if train_combo_metric else np.nan)
                test_values.append(np.mean(test_combo_metric) if test_combo_metric else np.nan)
            label = f"Train {species}-{genotype}"
            if not np.all(np.isnan(train_values)):
                ax.plot(image_size_list, train_values, marker='o', label=label, linestyle='-', color=colors[idx % len(colors)])
            label = f"Test {species}-{genotype}"
            if not np.all(np.isnan(test_values)):
                ax.plot(image_size_list, test_values, marker='s', label=label, linestyle='dashed', color=colors[idx % len(colors)])
        ax.set_xticks(image_size_list)
        ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
