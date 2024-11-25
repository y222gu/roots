import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import cv2
from ultralytics import YOLO
from skimage.measure import label, regionprops

def load_ground_truth_mask(annotation_file, image_shape):
    # Parse YOLO annotations and create a binary mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    height, width = image_shape[:2]
    
    with open(annotation_file, 'r') as file:
        for line in file:
            parts = line.strip().split()[1:]  # Skip class ID
            points = np.array(parts, dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= width
            points[:, 1] *= height
            points = points.astype(np.int32)
            cv2.fillPoly(mask, [points], 1)  # Fill polygon as 1
    return mask

def calculate_pixel_accuracy(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_pixels = gt_mask.sum()  # Total annotated pixels
    return intersection / total_pixels if total_pixels > 0 else 0

def calculate_pixel_sensitivity(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_gt_pixels = gt_mask.sum()  # Total ground truth object pixels
    return intersection / total_gt_pixels if total_gt_pixels > 0 else 0

def calculate_pixel_specificity(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    total_pred_pixels = pred_mask.sum()  # Total predicted object pixels
    return intersection / total_pred_pixels if total_pred_pixels > 0 else 0

def plot_confusion_matrix(gt_mask, pred_mask, save_path):
    # Flatten the masks to 1D arrays for confusion matrix calculation
    gt_mask_flat = gt_mask.flatten()
    pred_mask_flat = pred_mask.flatten()

    # Compute the confusion matrix
    cm = confusion_matrix(gt_mask_flat, pred_mask_flat, labels=[0, 1])

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Background', 'Object'], yticklabels=['Background', 'Object'])
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    
    # Save the plot with the image name and '_matrices'
    plt.savefig(save_path)
    plt.close()

def draw_masks(image, pred_mask, gt_mask, title):
    # Create an overlay with transparency
    overlay = image.copy()
    alpha = 0.5  # Transparency factor

    # Create masks
    overlap_mask = np.logical_and(pred_mask, gt_mask).astype(np.uint8)  # Overlap (green)
    pred_only_mask = np.logical_and(pred_mask, np.logical_not(overlap_mask)).astype(np.uint8)  # Predicted only (yellow)
    gt_only_mask = np.logical_and(gt_mask, np.logical_not(overlap_mask)).astype(np.uint8)  # Ground truth only (blue)


    # Convert binary masks to 3-channel
    pred_only_mask_color = np.stack([pred_only_mask * 255, pred_only_mask * 0, pred_only_mask * 0], axis=-1)  # Yellow
    gt_only_mask_color = np.stack([gt_only_mask * 0, gt_only_mask * 255, gt_only_mask * 255], axis=-1)  # Blue
    overlap_mask_color = np.stack([overlap_mask * 0, overlap_mask * 255, overlap_mask * 0], axis=-1)  # Green

    # Apply the masks with transparency
    overlay = cv2.addWeighted(overlay, 1, pred_only_mask_color, alpha, 0)
    overlay = cv2.addWeighted(overlay, 1, gt_only_mask_color, alpha, 0)
    overlay = cv2.addWeighted(overlay, 1, overlap_mask_color, alpha, 0)
    # Side-by-side comparison
    combined_image = np.hstack((overlay, image))

    # Dynamically scale legend size
    legend_height = int(combined_image.shape[0] * 0.1)  # Legend height is 10% of image height
    font_scale = combined_image.shape[0] / 800  # Font scale proportional to image size
    thickness = max(1, int(font_scale * 2))  # Scaled thickness
    spacing = int(legend_height * 0.2)  # Spacing between legend items

    # Create a blank overlay for the legend in the top-right corner
    legend_overlay = combined_image.copy()

    # Add legend items
    items = [
        ("Human Annotated (Yellow)", (0, 255, 255)),
        ("Model Predicted (Blue)", (255, 0, 0)),
        ("Overlap (Green)", (0, 255, 0)),
    ]

    x_start = spacing
    y_start = combined_image.shape[0]-spacing-int(legend_height*0.5)
    for text, color in items:
        # Draw color box
        cv2.rectangle(legend_overlay, (x_start, y_start), (x_start + int(legend_height * 0.4), y_start + int(legend_height * 0.4)), color, -1)
        # Add text next to the color box
        cv2.putText(legend_overlay, text, (x_start + int(legend_height * 0.5), y_start + int(legend_height * 0.35)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y_start -= int(legend_height * 0.5)

    # Add title at the top of the image
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, thickness)[0]
    text_x = (legend_overlay.shape[1] - text_size[0]) // 2
    text_y = text_size[1] + spacing
    cv2.putText(legend_overlay, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 1.5, (255, 255, 255), thickness, cv2.LINE_AA)

    return legend_overlay


def count_polygons(mask):
    """
    Count the number of distinct polygons in a binary mask.
    """
    labeled_mask = label(mask)  # Label connected regions
    return len(regionprops(labeled_mask))  # Count regions

def compare_polygon_counts(pred_mask, gt_mask):
    """
    Compare the number of polygons between predicted and ground truth masks.
    """
    pred_count = count_polygons(pred_mask)
    gt_count = count_polygons(gt_mask)
    difference = abs(pred_count - gt_count)
    return pred_count, gt_count, difference


if __name__ == '__main__':
    # Load the YOLO model
    model = YOLO(r'C:\Users\Yifei\Documents\roots\aerenchyma_segmentation\runs\yolov8_segmentation3\weights\best.pt')

    # Paths
    val_images_path = os.path.join(os.getcwd(), "aerenchyma_segmentation", "data_for_segmentation", "images", "val")
    labels_path = os.path.join(os.getcwd(), "aerenchyma_segmentation", "data_for_segmentation", "labels", "val")
    output_path = os.path.join(os.getcwd(), "aerenchyma_segmentation", "data_for_segmentation", "images", "val_predictions")
    os.makedirs(output_path, exist_ok=True)

# Process each image
for image_file in os.listdir(val_images_path):
    image_path = os.path.join(val_images_path, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        continue

    # Run YOLO inference
    results = model.predict(image, imgsz=1024, conf=0.2, task='segment', verbose=False)
    masks = results[0].masks

    # Load ground truth mask
    label_file = os.path.splitext(image_file)[0] + '.txt'
    annotations_path = os.path.join(labels_path, label_file)
    gt_mask = load_ground_truth_mask(annotations_path, image.shape)

    # Combine predicted masks
    pred_mask_combined = np.zeros_like(gt_mask)
    if masks is not None:
        for mask in masks.data:
            mask = (mask.cpu().numpy() > 0.5).astype(np.uint8)
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred_mask_combined = np.logical_or(pred_mask_combined, mask_resized).astype(np.uint8)

    # Count polygons in predicted and ground truth masks
    pred_count, gt_count, diff_count = compare_polygon_counts(pred_mask_combined, gt_mask)
    print(f"{image_file}: Predicted Polygons = {pred_count}, Ground Truth Polygons = {gt_count}, Difference = {diff_count}")

    # Calculate pixel sensitivity and specificity
    sensitivity = calculate_pixel_sensitivity(pred_mask_combined, gt_mask)
    specificity = calculate_pixel_specificity(pred_mask_combined, gt_mask)
    title = f"Pixel-level Sensitivity: {sensitivity:.2%} | Specificity: {specificity:.2%}"

    # Draw masks and create combined image
    combined_image = draw_masks(image, pred_mask_combined, gt_mask, title)

    # Save the combined image
    save_path = os.path.join(output_path, f"comparison_{image_file}")
    cv2.imwrite(save_path, combined_image)

    # Plot and save confusion matrix
    cm_save_path = os.path.join(output_path, f"{os.path.splitext(image_file)[0]}_matrices.png")
    plot_confusion_matrix(gt_mask, pred_mask_combined, cm_save_path)

