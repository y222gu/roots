import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from segmentation_models_pytorch.utils.metrics import IoU

def visualize_all_predictions_with_manual_annotation(aere_model, whole_root_area_model, channels, dataset, output_folder):
    """
    For each sample in the validation dataset:
      - Grabs the 3-channel image tensor & true mask
      - Predicts the mask with your model
      - Creates per-channel overlays (true vs pred)
      - Saves each channel comparison as a PNG with black background
      - Saves a CSV with summary statistics for each channel.
    """
    aere_model.eval()
    whole_root_area_model.eval()
    device   = next(aere_model.parameters()).device
    os.makedirs(output_folder, exist_ok=True)

    results = []
    for idx in range(len(dataset)):
        # --- get data & predict ---
        image, true_mask, sample_id, images_original = dataset.__getitem__(idx)
        image_tensor = image.unsqueeze(0).to(device)          # (1,3,H,W)

        with torch.no_grad():
            logits = aere_model(image_tensor)                      # (1,1,H,W)

        # 1) convert to probabilities
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

        # unique_vals = np.unique(pred_mask)
        # print("Unique values in mask:", unique_vals)

        # --- build RGB overlays ---
        def make_overlay(mask):
            # mask: 2D np array of ints {0,1}
            ov = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            ov[mask == 1] = [255, 0, 255]    # blue
            return ov

        true_ov = make_overlay(true_mask.numpy() if torch.is_tensor(true_mask) else true_mask)
        pred_ov = make_overlay(pred_mask)

        # if pred_whole_root_mask is None:
        #     pred_whole_root_mask = np.zeros_like(pred_mask, dtype=np.uint8)
        #     pred_whole_root_area = None
        # pred_whole_root_ov = np.zeros((pred_whole_root_mask.shape[0], pred_whole_root_mask.shape[1], 3), dtype=np.uint8)
        # pred_whole_root_ov[pred_whole_root_mask == 1] = [255, 0, 255]

        alpha = 0.5
        # --- build composite image with overlaid mask ---
        # Convert the 3-channel image tensor to a numpy array with shape (H, W, 3) in [0,255]
        composite_img = np.transpose(image.cpu().numpy(), (1, 2, 0))
        composite_img = (composite_img * 255).clip(0, 255).astype(np.uint8)
        # Apply gamma correction for better visibility
        inv_gamma = 0.2
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        composite_img = cv2.LUT(composite_img, table)

        # Blend the composite image with the mask overlay using alpha = 0.5
        blended = cv2.addWeighted(composite_img, 1 - alpha, pred_ov, alpha, 0)

        # Blend composite image with true annotation overlay and predicted mask overlay
        alpha = 0.5
        true_blended = cv2.addWeighted(composite_img, 1 - alpha, true_ov, alpha, 0)
        # blended is already computed for the prediction overlay
        
        # Plot the true annotation, predicted mask, and original image in three subplots
        fig, axs = plt.subplots(1, 3, figsize=(30, 10), facecolor='black')
        for ax in axs:
            ax.set_facecolor('black')
            ax.axis('off')
        
        axs[0].imshow(cv2.cvtColor(true_blended, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"True Annotation: Area {np.sum((true_mask.numpy() if torch.is_tensor(true_mask) else true_mask) == 1)}", color='white', fontsize=25)
        
        axs[1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Prediction: Area {np.sum(pred_mask == 1)}", color='white', fontsize=25)
        
        axs[2].imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
        axs[2].set_title("Original Image", color='white', fontsize=25)

        plt.tight_layout(pad=0.1, w_pad=0.1, h_pad=0.1)
        
        fig.suptitle(f"{sample_id} - True vs Predicted", color='white', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save the composite image
        save_path = os.path.join(output_folder, "prediction", f"{sample_id}_composite.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)

        # Calculate pixel-level metrics for logging
        pred_aere_area = np.sum(pred_mask == 1)
        true_mask_np = true_mask.numpy() if torch.is_tensor(true_mask) else true_mask
        true_aere_area = np.sum(true_mask_np == 1)

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
            'composite_predicted_area': pred_aere_area,
            'composite_true_area': true_aere_area,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy,
            'iou': iou
        })

        # After processing all samples, save the results to CSV.
        results_df = pd.DataFrame(results)
        csv_save_path = os.path.join(output_folder, "prediction", 'composite_mask_intensity_results.csv')
        os.makedirs(os.path.dirname(csv_save_path), exist_ok=True)
        results_df.to_csv(csv_save_path, index=False)

def visualize_all_predictions_without_manual_annotation(aere_model, whole_root_area_model, dataset, output_folder):
    """
    For each sample in the validation dataset:
      - Grabs the 3-channel image tensor & true mask
      - Predicts the mask with your model
      - Creates per-channel overlays (only pred)
      - Saves each channel comparison as a PNG with black background
        - Saves a CSV with summary statistics for each channel.
    """
    aere_model.eval()
    whole_root_area_model.eval()
    device   = next(aere_model.parameters()).device
    os.makedirs(output_folder, exist_ok=True)

    # export the result to a csv file
    import pandas as pd
    results = []

    for idx in range(len(dataset)):
        # --- get data & predict ---
        image, sample_id, images_original = dataset.__getitem__(idx)
        image_tensor = image.unsqueeze(0).to(device)          # (1,3,H,W)

        print('size of image tensor:', image_tensor.shape)
        print('size of image preprocessed:', image.shape)
        # pred_whole_root_mask, pred_whole_root_area = segment_root(whole_root_area_model, image_tensor, option='largest_area', confidence_threshold=0.01)

        with torch.no_grad():
            aere_logits = aere_model(image_tensor)                      # (1,1,H,W)

        # 1) convert to probabilities
        aere_probs = torch.sigmoid(aere_logits)
        aere_pred_mask = (aere_probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

        with torch.no_grad():
            pred_whole_root_logits = whole_root_area_model(image_tensor)  # (1,1,H,W)
        pred_whole_root_probs = torch.sigmoid(pred_whole_root_logits)
        pred_whole_root_mask = (pred_whole_root_probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)


        pred_ov_aere = np.zeros((aere_pred_mask.shape[0], aere_pred_mask.shape[1], 3), dtype=np.uint8)
        pred_ov_aere[aere_pred_mask == 1] = [255, 0, 0]    # blue

        pred_ov_whole_root = np.zeros((pred_whole_root_mask.shape[0], pred_whole_root_mask.shape[1], 3), dtype=np.uint8)
        pred_ov_whole_root[pred_whole_root_mask == 1] = [255, 0, 255]    # magenta

        alpha = 0.5

        # --- visualize combined channels ---
        # Create a composite color image from the three channels
        # Assuming image_preprocessed has shape (3, H, W)
        composite_img = np.transpose(image.cpu().numpy(), (1, 2, 0))
        composite_img = (composite_img * 255).clip(0, 255).astype(np.uint8)
        # Apply gamma correction for better visibility
        inv_gamma = 0.2
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        composite_img = cv2.LUT(composite_img, table)

        # Blend the composite image with the Aere prediction overlay and then the whole root overlay
        blended = cv2.addWeighted(composite_img, 1 - alpha, pred_ov_whole_root, alpha, 0)
        blended = cv2.addWeighted(blended, 1 - alpha, pred_ov_aere, alpha, 0)


        # Calculate total areas and the ratio
        pred_aere_area = np.sum(aere_pred_mask == 1)
        pred_whole_root_area = np.sum(pred_whole_root_mask == 1)
        if pred_whole_root_area is not None and pred_whole_root_area > 0:
            aere_whole_root_ratio = pred_aere_area / pred_whole_root_area
        else:
            aere_whole_root_ratio = None

        results.append({
            'sample_id': sample_id,
            'predicted_aere_area': pred_aere_area,
            'predicted_whole_root_area': pred_whole_root_area,
            'predicted_aere_over_whole_root_ratio': aere_whole_root_ratio,
        })

        # Plot the original composite image and blended image on a black background with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(20, 10), facecolor='black')
        for ax in axs:
            ax.set_facecolor('black')
            ax.axis('off')
        # Display the original image without overlay
        axs[0].imshow(cv2.cvtColor(composite_img, cv2.COLOR_BGR2RGB))
        axs[0].set_title("Original Image", color='white', fontsize=16)
        # Display the blended image with mask overlay
        axs[1].imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        axs[1].set_title(f"Mask area: {pred_aere_area}\nAere/Whole root ratio: {aere_whole_root_ratio}", color='white', fontsize=16)
        fig.suptitle(f"{sample_id} Predicted Masks with Binary Models", color='white', fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = os.path.join(output_folder, "prediction", f"{sample_id}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight')
        plt.close(fig)

    # Save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_folder, "prediction", 'mask_intensity_results_with_binary_models.csv'), index=False)