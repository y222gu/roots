import os
import cv2
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# last updated November 10, 2024 by Lucas DeMello (Brady Lab)

# Folder paths for normalization and prediction
folder1 = r"C:\Users\Root Project\Pictures\test3\pics norm"  # Folder for calculating target mean brightness
folder2 = r"C:\Users\Root Project\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\20240715-19_automated_3-plates\plate_2_processed\DAPI_cropped"  # Folder to normalize
output_folder_outer = r"C:\Users\Root Project\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\20240715-19_automated_3-plates\plate_2_processed\DAPI_cropped_outer"
output_folder_inner = r"C:\Users\Root Project\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\20240715-19_automated_3-plates\plate_2_processed\DAPI_cropped_inner"

# Model paths
model_path_outer = r'C:\Users\Root Project\Documents\maskrcnn1\training runs\run 3 8e\maskrcnnweights.pth'
model_path_inner = r'C:\Users\Root Project\Documents\maskrcnn1\training runs\run 50 10e\maskrcnnweights.pth'

# Normalization function
def normalize_brightness(img_array, target_mean):
    img_float = img_array.astype(float)
    current_mean = np.mean(img_float)
    scale_factor = target_mean / current_mean if current_mean > 0 else 1
    img_normalized = img_float * scale_factor
    return np.clip(img_normalized, 0, 65535).astype(np.uint16)

# Calculate the mean brightness of images in folder1
def calculate_mean_brightness(folder):
    image_extensions = ('.tif', '.png', '.jpg', '.jpeg')
    images = [img for img in os.listdir(folder) if img.lower().endswith(image_extensions)]
    total_brightness = 0
    total_pixels = 0
    for image_name in images:
        img_path = os.path.join(folder, image_name)
        img = Image.open(img_path)
        img_array = np.array(img).astype(float)
        total_brightness += np.sum(img_array)
        total_pixels += img_array.size
    return total_brightness / total_pixels if total_pixels > 0 else 0

# Function to normalize images in folder2 based on folder1's average brightness
def normalize_images(folder1, folder2):
    mean_folder1 = calculate_mean_brightness(folder1)
    normalized_images = []
    image_extensions = ('.tif', '.png', '.jpg', '.jpeg')
    images = [img for img in os.listdir(folder2) if img.lower().endswith(image_extensions)]

    for image_name in images:
        img_path = os.path.join(folder2, image_name)
        img = Image.open(img_path)
        img_array = np.array(img)
        normalized_img_array = normalize_brightness(img_array, mean_folder1)
        normalized_images.append((image_name, normalized_img_array))

    return normalized_images

# Model setup
def get_instance_segmentation_model(num_classes, model_weights_path):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                                                               hidden_layer,
                                                                                               num_classes)
    model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    return model

def predict(model, image_array, device):
    # Normalize image
    img_normalized = (image_array / 256).astype('uint8')

    # Check if the image is grayscale or RGB
    if len(img_normalized.shape) == 2:  # Grayscale image
        image_rgb = Image.fromarray(img_normalized, mode='L').convert('RGB')
    elif len(img_normalized.shape) == 3 and img_normalized.shape[2] == 3:  # RGB image
        image_rgb = Image.fromarray(img_normalized)
    else:
        raise ValueError("Unsupported image format: Expected grayscale or RGB.")

    # Transform to tensor
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image_rgb).to(device)

    # Prediction
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    
    return prediction
def process_predictions(prediction, image_array, output_folder, filename):
    if len(prediction['masks']) == 0:
        return

    combined_mask = torch.zeros_like(prediction['masks'][0, 0])
    individual_masks = []
    for mask, score in zip(prediction['masks'], prediction['scores']):
        if score > 0.1:
            individual_masks.append(mask[0] > 0.5)
            combined_mask = torch.logical_or(combined_mask, mask[0] > 0.5)

    combined_mask = combined_mask.cpu().numpy()
    individual_masks = [mask.cpu().numpy() for mask in individual_masks]
    
    # Filter out small masks
    filtered_masks = [mask for mask in individual_masks if np.sum(mask) > 1000]
    
    combined_filtered_mask = np.zeros_like(combined_mask, dtype=np.uint8)
    for mask in filtered_masks:
        combined_filtered_mask = np.maximum(combined_filtered_mask, mask.astype(np.uint8))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_filtered_mask, connectivity=8)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 0
    largest_mask = (labels == largest_label).astype(np.uint8)
    
    # Apply the mask to the image (for each channel)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:  # RGB image
        masked_image = np.zeros_like(image_array)
        for i in range(3):  # Apply the mask to each channel
            masked_image[..., i] = np.where(largest_mask == 1, image_array[..., i], 0)
    else:  # Grayscale image (already 2D)
        masked_image = np.where(largest_mask == 1, image_array, 0)

    base_name = os.path.splitext(filename)[0]
    sub_folder = os.path.join(output_folder, base_name)
    os.makedirs(sub_folder, exist_ok=True)

    masked_image = Image.fromarray(masked_image.astype(np.uint16))
    masked_image.save(os.path.join(sub_folder, f"{base_name}_masked.tif"))

    np.save(os.path.join(sub_folder, f"{base_name}_mask.npy"), largest_mask)
    
    plt.figure(figsize=(15, 5 * 2))
    plt.subplot(2, 1, 1)
    plt.imshow(image_array, cmap='gray' if image_array.ndim == 2 else None)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.imshow(masked_image, cmap='gray' if image_array.ndim == 2 else None)
    plt.title('Masked Image')
    plt.axis('off')

    plt.savefig(os.path.join(sub_folder, f"{base_name}_visualization.png"))
    plt.close()


# Main processing function
def main():
    # Load models
    model_outer = get_instance_segmentation_model(2, model_path_outer)
    model_inner = get_instance_segmentation_model(2, model_path_inner)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_outer.to(device)
    model_inner.to(device)

    # Normalize images
    normalized_images = normalize_images(folder1, folder2)

    # Process predictions
    for image_name, normalized_img_array in normalized_images:
        prediction_outer = predict(model_outer, normalized_img_array, device)
        process_predictions(prediction_outer, normalized_img_array, output_folder_outer, image_name)

        prediction_inner = predict(model_inner, normalized_img_array, device)
        process_predictions(prediction_inner, normalized_img_array, output_folder_inner, image_name)

if __name__ == "__main__":
    main()
