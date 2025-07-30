import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

def visualize_all_predictions_with_manual_annotation(model, dataset, output_folder, alpha=0.5):
    """
    For each sample in `dataset`:
      - run model → 2-ch logits → sigmoid → preds [2,H,W]
      - pull true_mask [2,H,W]
      - compute: areas, ratios, precision, recall, accuracy, IoU per class
      - save CSV of metrics
      - make one figure per sample: [ True overlay | Pred overlay ]
          on the original 3-ch image (RGB composite),
          whole=magenta, part=blue.
    """
    model.eval()
    device = next(model.parameters()).device
    os.makedirs(output_folder, exist_ok=True)
    metrics = []
    class_names = ["whole", "part"]

    for idx in range(len(dataset)):
        # --- 1. pull sample & predict ---
        image, true_mask, sample_id = dataset[idx]
        # image: torch.Tensor [3,H,W], true_mask: [2,H,W]
        x = image.unsqueeze(0).to(device)   # [1,3,H,W]
        with torch.no_grad():
            logits = model(x)               # [1,2,H,W]
        probs = torch.sigmoid(logits)[0]    # [2,H,W]
        preds = (probs > 0.5).cpu().numpy().astype(np.uint8)

        # ensure true_mask is numpy array
        if torch.is_tensor(true_mask):
            true_np = true_mask.cpu().numpy().astype(np.uint8)
        else:
            true_np = true_mask.astype(np.uint8)   # [2,H,W]

        # --- 2. areas & ratios ---
        pred_whole_root = int(preds[0].sum())
        pred_aere  = int(preds[1].sum())
        true_whole_root = int(true_np[0].sum())
        true_aere  = int(true_np[1].sum())

        pred_ratio = pred_aere / (pred_whole_root + 1e-6)
        true_ratio = true_aere  / (true_whole_root  + 1e-6)

        # --- 3. per-class metrics ---
        for c, name in enumerate(class_names):
            p = preds[c].flatten()
            t = true_np[c].flatten()
            TP = int(((p==1) & (t==1)).sum())
            FP = int(((p==1) & (t==0)).sum())
            FN = int(((p==0) & (t==1)).sum())
            TN = int(((p==0) & (t==0)).sum())

            precision = TP / (TP + FP + 1e-6)
            recall    = TP / (TP + FN + 1e-6)
            accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-6)
            iou       = TP / (TP + FP + FN + 1e-6)

            metrics.append({
                "sample_id": sample_id,
                "class":     name,
                "pred_area": pred_whole_root if c==0 else pred_aere,
                "true_area": true_whole_root if c==0 else true_aere,
                "pred_whole_root_area": pred_whole_root,
                "pred_aere_area":  pred_aere,
                "true_whole_root_area": true_whole_root,
                "true_aere_area":  true_aere,
                "pred_aere/whole_root_ratio": pred_ratio,
                "true_aere/whole_root_ratio": true_ratio,
                "precision": precision,
                "recall":    recall,
                "accuracy":  accuracy,
                "iou":       iou
            })

        # --- 4. build RGB composite and overlays ---
        # to numpy H×W×3 in [0..255]
        img_np = image.permute(1, 2, 0).cpu().numpy()
        img_uint8 = (255 * (img_np - img_np.min()) /
                     (img_np.max() - img_np.min() + 1e-6)).astype(np.uint8)

        # true overlay
        overlay_true = np.zeros_like(img_uint8)
        overlay_true[ true_np[0]==1 ] = [255,   0, 255]  # magenta
        overlay_true[ true_np[1]==1 ] = [  0,   0, 255]  # blue
        true_blend   = cv2.addWeighted(img_uint8, 1-alpha,
                                       overlay_true,  alpha, 0)

        # pred overlay
        overlay_pred = np.zeros_like(img_uint8)
        overlay_pred[ preds[0]==1 ] = [255,   0, 255]
        overlay_pred[ preds[1]==1 ] = [  0,   0, 255]
        pred_blend   = cv2.addWeighted(img_uint8, 1-alpha,
                                       overlay_pred,  alpha, 0)

        # --- 5. plot & save ---
        # Create a figure with black background and three subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
        for ax in axes:
            ax.axis('off')
            ax.set_facecolor('black')

        # Original image without overlay
        axes[0].imshow(img_uint8)
        axes[0].set_title("Original image", color='white', fontsize=25)

        # Ground-truth overlay with ratio
        axes[1].imshow(true_blend)
        axes[1].set_title(f"Ground-truth overlay (ratio: {true_ratio*100:.1f}%)", color='white', fontsize=25)

        # Prediction overlay with ratio
        axes[2].imshow(pred_blend)
        axes[2].set_title("Prediction overlay (ratio: {:.1f}%)".format(pred_ratio * 100), color='white', fontsize=25)
        
        plt.suptitle(sample_id, color='white', fontsize=25)
        plt.tight_layout(pad=0.1)  # very tight layout
        
        out_png = os.path.join(output_folder, f"{sample_id}.png")
        plt.savefig(out_png, dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)

    # --- 6. dump metrics to CSV ---
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(output_folder, "mask_intensity_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved validation metrics → {csv_path}")


def visualize_all_predictions_without_manual_annotation(model, dataset, output_folder, alpha=0.5):
    """
    For each sample in `dataset`:
      - run model → 2-ch logits → sigmoid → preds [2,H,W]
      - pull true_mask [2,H,W]
      - compute: areas, ratios, precision, recall, accuracy, IoU per class
      - save CSV of metrics
      - make one figure per sample: [ True overlay | Pred overlay ]
          on the original 3-ch image (RGB composite),
          whole=magenta, part=blue.
    """
    model.eval()
    device = next(model.parameters()).device
    os.makedirs(output_folder, exist_ok=True)
    results = []
    class_names = ["whole", "part"]

    for idx in range(len(dataset)):
        # --- 1. pull sample & predict ---
        image, sample_id = dataset[idx]
        # image: torch.Tensor [3,H,W], true_mask: [2,H,W]
        x = image.unsqueeze(0).to(device)   # [1,3,H,W]
        with torch.no_grad():
            logits = model(x)               # [1,2,H,W]
        probs = torch.sigmoid(logits)[0]    # [2,H,W]
        preds = (probs > 0.5).cpu().numpy().astype(np.uint8)

        # --- 2. areas & ratios ---
        pred_whole_root = int(preds[0].sum())
        pred_aere  = int(preds[1].sum())
        pred_ratio = pred_aere / (pred_whole_root + 1e-6)

        # --- 3. per-class metrics ---
        for c, name in enumerate(class_names):
            p = preds[c].flatten()


            results.append({
                "sample_id": sample_id,
                "predicted_aere_area":  pred_aere,
                "predicted_whole_root_area": pred_whole_root,
                "predicted_aere_over_whole_root_ratio": pred_ratio,
            })

        # --- 4. build RGB composite and overlays ---
        # to numpy H×W×3 in [0..255]
        # image = image.permute(1, 2, 0).cpu().numpy()
        # image = (255 * (image - image.min()) /
        #              (image.max() - image.min() + 1e-6)).astype(np.uint8)
        
        image = np.transpose(image.cpu().numpy(), (1, 2, 0))
        image = (image * 255).clip(0, 255).astype(np.uint8)
        # Apply gamma correction for better visibility
        inv_gamma = 0.2
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        image = cv2.LUT(image, table)

        # pred overlay
        overlay_pred = np.zeros_like(image)
        overlay_pred[ preds[0]==1 ] = [255,   0, 255]
        overlay_pred[ preds[1]==1 ] = [  0,   0, 255]
        pred_blend   = cv2.addWeighted(image, 1-alpha,
                                       overlay_pred,  alpha, 0)

        # --- 5. plot & save ---
        # Create a figure with black background and three subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 6), facecolor='black')
        for ax in axes:
            ax.axis('off')
            ax.set_facecolor('black')

        # Original image without overlay
        axes[0].imshow(image)
        axes[0].set_title("Original image", color='white', fontsize=25)

        # Prediction overlay with ratio
        axes[1].imshow(pred_blend)
        axes[1].set_title(f"Mask area: {pred_aere}\nAere/Whole root ratio: {pred_ratio}", color='white', fontsize=16)
        fig.suptitle(f"{sample_id} Predicted Masks with Multi-labels Model", color='white', fontsize=18)
        plt.tight_layout(pad=0.1)  # very tight layout
        
        save_path = os.path.join(output_folder, "prediction", f"{sample_id}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, "prediction", 'mask_intensity_results_with_multi_labels_model.csv'), index=False)