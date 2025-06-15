import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from segmentation_models_pytorch.utils.metrics import IoU

def visualize_all_predictions_with_manual_annotation(model, channels, dataset, output_folder):
    """
    For each sample in the validation dataset:
      - Grabs the 3-channel image tensor & true mask
      - Predicts the mask with your model
      - Creates per-channel overlays (true vs pred)
      - Saves each channel comparison as a PNG with black background
      - Saves a CSV with summary statistics for each channel.
    """
    model.eval()
    device   = next(model.parameters()).device
    os.makedirs(output_folder, exist_ok=True)

    results = []
    for idx in range(len(dataset)):
        # --- get data & predict ---
        image, true_mask, sample_id = dataset.__getitem__(idx)

        image_tensor = image.unsqueeze(0).to(device)          # (1,3,H,W)
        with torch.no_grad():
            logits = model(image_tensor)                      # (1,1,H,W)

        # 1) convert to probabilities
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

        unique_vals = np.unique(pred_mask)
        print("Unique values in mask:", unique_vals)

        # --- build RGB overlays ---
        def make_overlay(mask):
            # mask: 2D np array of ints {0,1}
            ov = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            ov[mask == 1] = [255, 0, 0]    # blue
            return ov

        true_ov = make_overlay(true_mask.numpy() if torch.is_tensor(true_mask) else true_mask)
        pred_ov = make_overlay(pred_mask)

        alpha = 0.5

        # --- visualize per channel ---
        for i, ch in enumerate(channels):
            # 1) pull out the ith channel as numpy float in [0,1]
            chan = image[i].cpu().numpy()
            # 2) scale back to [0,255] and cast to uint8
            chan_uint8 = (chan * 255).clip(0, 255).astype(np.uint8)
            # 3) gray→BGR so it has 3 channels
            channel_img = cv2.cvtColor(chan_uint8, cv2.COLOR_GRAY2BGR)

            # 4) blend—both are uint8 now
            true_blend = cv2.addWeighted(channel_img, 1-alpha, true_ov, alpha, 0)
            pred_blend = cv2.addWeighted(channel_img, 1-alpha, pred_ov, alpha, 0)

            # 5)
            # total area of the predicted mask
            predicted_total_area = np.sum(pred_mask == 1)
            # total intensity of the predicted mask
            predicted_total_intensity = np.sum(chan_uint8[pred_mask == 1])
            # extract the intensity values for the predicted mask in each channel
            if np.sum(pred_mask == 1) > 0:
                predicted_channel_intensity = round(np.mean(chan_uint8[pred_mask == 1]))
            else:
                predicted_channel_intensity = None

            # total area of the true mask
            true_total_area = np.sum(true_mask.numpy() == 1)
            # total intensity of the true mask
            true_total_intensity = np.sum(chan_uint8[true_mask.numpy() == 1])
            # extract the intensity values for the true mask in each channel
            if np.sum(true_mask.numpy() == 1) > 0:
                true_channel_intensity = round(np.mean(chan_uint8[true_mask.numpy() == 1]))
            else:
                true_channel_intensity = None

            # Calculate pixel-level metrics: precision, recall, accuracy, and IoU
            true_mask_np = true_mask.numpy() if torch.is_tensor(true_mask) else true_mask
            pred_flat = pred_mask.flatten()
            true_flat = true_mask_np.flatten()
            TP = np.sum((pred_flat == 1) & (true_flat == 1))
            FP = np.sum((pred_flat == 1) & (true_flat == 0))
            FN = np.sum((pred_flat == 0) & (true_flat == 1))
            TN = np.sum((pred_flat == 0) & (true_flat == 0))
            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)
            accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
            pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float()
            true_tensor = torch.from_numpy(true_mask_np).unsqueeze(0).unsqueeze(0).float()
            iou = IoU(threshold=0.5)(pred_tensor, true_tensor).item()

            results.append({
                'sample_id': sample_id,
                'channel': ch,
                'predicted_total_area': predicted_total_area,
                'predicted_total_intensity': predicted_total_intensity,
                'predicted_channel_intensity': predicted_channel_intensity,
                'true_total_area': true_total_area,
                'true_total_intensity': true_total_intensity,
                'true_channel_intensity': true_channel_intensity,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'iou': iou
            })

            # 6) plot on black bg
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='black')
            for ax in axes:
                ax.set_facecolor('black')
                ax.axis('off')

            axes[0].imshow(cv2.cvtColor(true_blend, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f"True mask avg intensity: {true_channel_intensity}", color='white')
            axes[1].imshow(cv2.cvtColor(pred_blend, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f"Pred mask avg intensity: {predicted_channel_intensity}", color='white')

            fig.suptitle(f"{sample_id} — {ch}", color='white', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            save_path = os.path.join(output_folder,"prediction", f"{sample_id}_{ch}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
    
    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, "prediction",'mask_intensity_results.csv'), index=False)

def visualize_all_predictions_without_manual_annotation(model,channels, dataset, output_folder):
    """
    For each sample in the validation dataset:
      - Grabs the 3-channel image tensor & true mask
      - Predicts the mask with your model
      - Creates per-channel overlays (only pred)
      - Saves each channel comparison as a PNG with black background
        - Saves a CSV with summary statistics for each channel.
    """
    model.eval()
    device   = next(model.parameters()).device
    os.makedirs(output_folder, exist_ok=True)

    # export the result to a csv file
    import pandas as pd
    results = []

    for idx in range(len(dataset)):
        # --- get data & predict ---
        image, sample_id = dataset.__getitem__(idx)
        image_tensor = image.unsqueeze(0).to(device)          # (1,3,H,W)
    
        with torch.no_grad():
            logits = model(image_tensor)                      # (1,1,H,W)

        # 1) convert to probabilities
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

        # --- build RGB overlays ---
        def make_overlay(mask):
            # mask: 2D np array of ints {0,1}
            ov = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            ov[mask == 1] = [255, 0, 0]    # blue
            return ov

        pred_ov = make_overlay(pred_mask)

        alpha = 0.5

        # --- visualize per channel ---
        for i, ch in enumerate(channels):
            # rescale the ith channel to [0,255] and convert to uint8
            chan = image[i].cpu().numpy()
            chan_uint8 = (chan * 255).clip(0, 255).astype(np.uint8)
            channel_img = cv2.cvtColor(chan_uint8, cv2.COLOR_GRAY2BGR)
            pred_blend = cv2.addWeighted(channel_img, 1-alpha, pred_ov, alpha, 0)

            # total area of the predicted mask
            total_area = np.sum(pred_mask == 1)
            # total intensity of the predicted mask
            total_intensity = np.sum(chan_uint8[pred_mask == 1])
            # extract the intensity values for the predicted mask in each channel
            if np.sum(pred_mask == 1) > 0:
                channel_intensity = round(np.mean(chan_uint8[pred_mask == 1]))
            else:
                channel_intensity = None

            results.append({
                'sample_id': sample_id,
                'channel': ch,
                'total_area': total_area,
                'total_intensity': total_intensity,
                'channel_intensity': channel_intensity
            })

            # plot on black bg
            fig, ax = plt.subplots(figsize=(10, 4), facecolor='black')
            ax.set_facecolor('black')
            ax.axis('off')
            ax.imshow(cv2.cvtColor(pred_blend, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Mask area:{total_area}|Average intensity:{channel_intensity}", color='white')
            fig.suptitle(f"{sample_id}_{ch} Predicted", color='white', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])

            save_path = os.path.join(output_folder, "prediction", f"{sample_id}_{ch}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, "prediction", 'mask_intensity_results.csv'), index=False)