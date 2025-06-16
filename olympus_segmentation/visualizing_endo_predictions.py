import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from segmentation_models_pytorch.utils.metrics import IoU
from scipy.stats import pearsonr

def visualize_endo_predictions(model, channels, dataset, output_folder, manual_annotation='True', alpha = 0.5):
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
        # --- get data ---
        if manual_annotation == 'True':
            image, true_mask, sample_id, image_original = dataset.__getitem__(idx)
            # true mask
            true_mask = true_mask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            if true_mask.shape != image_original.shape:
                true_mask = cv2.resize(true_mask, (image_original.shape[1], image_original.shape[0]), interpolation=cv2.INTER_NEAREST)
            true_ov = make_overlay(true_mask)
        else:
            image, sample_id, image_original = dataset.__getitem__(idx)

        # predict endo mask
        image = image.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(image)
        # convert to probabilities
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        if pred_mask.shape != image_original.shape:
            pred_mask = cv2.resize(pred_mask, (image_original.shape[1], image_original.shape[0]), interpolation=cv2.INTER_NEAREST)
        pred_ov = make_overlay(pred_mask)

        # --- visualize per channel ---
        for i, ch in enumerate(channels):
            # 1) pull out the ith channel as numpy float in [0,1]
            chan = image_original[:,:,i]

            # 2) total area and intensity of the predicted mask
            predicted_total_area = np.sum(pred_mask == 1)
            predicted_total_intensity = np.sum(chan[pred_mask == 1])
            if np.sum(pred_mask == 1) > 0:
                predicted_average_intensity = round(np.mean(chan[pred_mask == 1]))
            else:
                predicted_average_intensity = None

            if manual_annotation == 'True':
                true_total_area = np.sum(true_mask == 1)
                # total intensity of the true mask
                true_total_intensity = np.sum(chan[true_mask == 1])
                # extract the intensity values for the true mask in each channel
                if np.sum(true_mask == 1) > 0:
                    true_average_intensity = round(np.mean(chan[true_mask == 1]))
                else:
                    true_average_intensity = None

                # Calculate pixel-level metrics: precision, recall, accuracy, and IoU
                pred_flat = pred_mask.flatten()
                true_flat = true_mask.flatten()
                TP = np.sum((pred_flat == 1) & (true_flat == 1))
                FP = np.sum((pred_flat == 1) & (true_flat == 0))
                FN = np.sum((pred_flat == 0) & (true_flat == 1))
                TN = np.sum((pred_flat == 0) & (true_flat == 0))
                precision = TP / (TP + FP + 1e-6)
                recall = TP / (TP + FN + 1e-6)
                accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
                pred_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float()
                true_tensor = torch.from_numpy(true_mask).unsqueeze(0).unsqueeze(0).float()
                iou = IoU(threshold=0.5)(pred_tensor, true_tensor).item()

                results.append({
                    'sample_id': sample_id,
                    'channel': ch,
                    'predicted_total_area': predicted_total_area,
                    'predicted_total_intensity': predicted_total_intensity,
                    'predicted_average_intensity': predicted_average_intensity,
                    'true_total_area': true_total_area,
                    'true_total_intensity': true_total_intensity,
                    'true_average_intensity': true_average_intensity,
                    'precision': precision,
                    'recall': recall,
                    'accuracy': accuracy,
                    'iou': iou
                })
            else:
                results.append({
                    'sample_id': sample_id,
                    'channel': ch,
                    'predicted_total_area': predicted_total_area,
                    'predicted_total_intensity': predicted_total_intensity,
                    'predicted_average_intensity': predicted_average_intensity,
                })


            channel_img = cv2.cvtColor(chan, cv2.COLOR_GRAY2BGR)
            pred_blend = cv2.addWeighted(channel_img, 1-alpha, pred_ov, alpha, 0)

            if manual_annotation == 'True':
                true_blend = cv2.addWeighted(channel_img, 1-alpha, true_ov, alpha, 0)
                fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor='black')
                for ax in axes:
                    ax.set_facecolor('black')
                    ax.axis('off')
                axes[0].imshow(cv2.cvtColor(true_blend, cv2.COLOR_BGR2RGB))
                axes[0].set_title(f"True mask avg intensity: {true_average_intensity}", color='white')
                axes[1].imshow(cv2.cvtColor(pred_blend, cv2.COLOR_BGR2RGB))
                axes[1].set_title(f"Pred mask avg intensity: {predicted_average_intensity}", color='white')

            else:
                fig, ax = plt.subplots(figsize=(10, 4), facecolor='black')
                ax.set_facecolor('black')
                ax.axis('off')
                ax.imshow(cv2.cvtColor(pred_blend, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Mask area:{predicted_total_area}|Average intensity:{predicted_average_intensity}", color='white')

            fig.suptitle(f"{sample_id}_{ch}", color='white', fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            save_path = os.path.join(output_folder, "prediction", f"{sample_id}_{ch}.png")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
            plt.close(fig)
    
    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, "prediction",'mask_intensity_results.csv'), index=False)

    # plot true average intensity vs predicted average intensity and fit  a line 
    if manual_annotation == 'True':
        plot_df = results_df.dropna(subset=['true_average_intensity', 'predicted_average_intensity'])
        if not plot_df.empty:
            true_vals = plot_df['true_average_intensity'].astype(float)
            pred_vals = plot_df['predicted_average_intensity'].astype(float)
            r, p_value = pearsonr(true_vals, pred_vals)
            print(f"Pearson r = {r:.3f}, p = {p_value:.3e}")

            plt.figure(figsize=(8, 6))
            plt.scatter(true_vals, pred_vals, c='blue', label='Data points')
            plt.plot([0,255], [0,255], "--", c='k', linewidth=1) 
            plt.ylim(0, true_vals.max() * 1.1)
            plt.xlim(0, pred_vals.max() * 1.1)
            
            # Fit a line to the data points if more than one point exists
            if len(true_vals) > 1:
                slope, intercept = np.polyfit(true_vals, pred_vals, 1)
                x_fit = np.array([true_vals.min(), true_vals.max()])
                y_fit = slope * x_fit + intercept
                plt.plot(x_fit, y_fit, color='red', label=f'Fit line: y={slope:.2f}x+{intercept:.2f}')
            
            # add pearson correlation coefficient to the plot
            plt.text(0.05, 0.95, f'Pearson r = {r:.3f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='k')
            plt.text(0.05, 0.90, f'p-value = {p_value:.2e}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', color='k')
            plt.xlabel('True Average Intensity')
            plt.ylabel('Predicted Average Intensity')
            plt.title('True vs Predicted Average Intensity')
            plt.legend()
            plt.tight_layout()
            overlay_path = os.path.join(output_folder, "prediction", "true_predicted_correlation_fit.png")
            plt.savefig(overlay_path, bbox_inches='tight')
            plt.close()

# --- build RGB overlays ---
def make_overlay(mask):
    # mask: 2D np array of ints {0,1}
    ov = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    ov[mask == 1] = [255, 0, 0]    # blue
    return ov