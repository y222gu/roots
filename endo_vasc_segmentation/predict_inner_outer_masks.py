import os
import cv2
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights


def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask,
                                                                                               hidden_layer,
                                                                                               num_classes)
    return model

def load_model(model_path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_instance_segmentation_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict(model, image_path, device):
    # Open the 16-bit grayscale image
    image = Image.open(image_path).convert("I;16")
        
    # Convert to a NumPy array
    img_array = np.array(image)
        
    # Normalize and convert to 8-bit
    img_normalized = cv2.normalize(img_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_rgb = Image.fromarray(img_normalized, mode='L').convert('RGB')  # Convert to RGB for prediction

    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image_rgb).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model([image_tensor])[0]
    return img_array, prediction  # Return the original array and the prediction

def process_images(threshold, model, input_folder, output_folder, device, min_mask_size=1000):
    os.makedirs(output_folder, exist_ok=True)
    run_marker = True
    num = 1

    for filename in os.listdir(input_folder):
        print(f"Processing file: {filename}")
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            image_path = os.path.join(input_folder, filename)
            image, prediction = predict(model, image_path, device)
            
            if len(prediction['masks']) == 0:
                print(f"No masks found for {filename}, skipping.")
                continue
            
            # Combine all masks with score above threshold
            combined_mask = torch.zeros_like(prediction['masks'][0, 0])
            individual_masks = []
            for mask, score in zip(prediction['masks'], prediction['scores']):
                if score > threshold:
                    individual_masks.append(mask[0] > 0.5)
                    combined_mask = torch.logical_or(combined_mask, mask[0] > 0.5)
            
            if len(individual_masks) == 0:
                print(f"No masks with score above {threshold} for {filename}, skipping.")
                continue

            combined_mask = combined_mask.cpu().numpy()
            individual_masks = [mask.cpu().numpy() for mask in individual_masks]
            
            # Visualize individual masks
            num_masks = len(individual_masks)
            if num_masks > 0:
                fig, axes = plt.subplots(1, num_masks, figsize=(15, 5))
                if num_masks == 1:
                    axes = [axes]  # Make it iterable if there's only one mask
                for i, mask in enumerate(individual_masks):
                    axes[i].imshow(mask, cmap='gray')
                    axes[i].set_title(f'Mask {i + 1}')
                    axes[i].axis('off')
                # plt.show()

            # Filter out small masks
            filtered_masks = []
            for mask in individual_masks:
                mask_area = np.sum(mask)
                print(f"Mask area: {mask_area}")
                if mask_area > min_mask_size:
                    filtered_masks.append(mask)
            
            # If no masks are left, skip this image
            if not filtered_masks:
                print(f"No masks larger than {min_mask_size} pixels for {filename}, skipping.")
                continue
            
            # Combine all filtered masks
            combined_filtered_mask = np.zeros_like(combined_mask, dtype=np.uint8)
            for mask in filtered_masks:
                combined_filtered_mask = np.maximum(combined_filtered_mask, mask.astype(np.uint8))
            
            # # Find the largest connected component in the combined filtered mask
            # num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined_filtered_mask, connectivity=8)
            # if num_labels > 1:
            #     largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            #     largest_mask = (labels == largest_label).astype(np.uint8)
            # else:
            #     largest_mask = combined_filtered_mask  # In case there's only one component
            
            # Resize the mask to match the original image dimensions
            resized_mask = cv2.resize(combined_filtered_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            
            # Apply the largest mask to the image
            masked_image = apply_mask(image, resized_mask)
            
            # Save results
            base_name = os.path.splitext(filename)[0]
            
            # Save the masked image in TIFF format
            masked_image.save(os.path.join(output_folder, f"{base_name}_masked.tif"), format="TIFF")
            # # save the mask
            # np.save(os.path.join(output_folder, f"{base_name}_mask.npy"), resized_mask)

            print("Image size:", image.shape)
            print("Mask size:", resized_mask.shape)
            print("Masked image size:",masked_image.size)

            # np.save(os.path.join(extra_mask_save_folder, f"{base_name}_mask.npy"), largest_mask)
            
            #  # Visualize and save
            # plt.figure(figsize=(15, 5 * 2))
            # plt.subplot(2, 1, 1)
            # plt.imshow(image, cmap='gray')  # Assuming it's a grayscale image
            # plt.title('Original Image')
            # plt.axis('off')
            
            # plt.subplot(2, 1, 2)
            # plt.imshow(masked_image, cmap='gray')  # Assuming it's a grayscale image
            # plt.title('Masked Image')
            # plt.axis('off')
            
            # plt.savefig(os.path.join(output_folder, f"{base_name}_visualization.png"))
            # plt.close()
            
        else:
            print("doesn't end in appropriate extension")

def apply_mask(image, mask):
    # Check if the image is already in NumPy format
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Ensure the mask is 2D
    if mask.ndim == 3:
        mask = mask[:,:,0]
    
    # Ensure the image is 2D
    if image.ndim == 3:
        image = image[:,:,0]
    
    # Apply the mask to the image
    masked_image = np.where(mask, image, 0)

    # Convert back to image format
    return Image.fromarray(masked_image.astype(np.uint16))


def predict_inner_outer_masks(input_folder):

    # # inner endo
    inner_model_path = r".\endo_vasc_segmentation\weights\inner_maskrcnnweights.pth"  # path to your saved model
    inner_num_classes = 2
    inner_input_folder = os.path.join(input_folder, 'DAPI_cropped_normalized')
    inner_output_folder = os.path.join(input_folder, 'inner_masks')

    model, device = load_model(inner_model_path, inner_num_classes)
    process_images(0.1, model, inner_input_folder, inner_output_folder, device)

    # # outer endo
    outer_model_path = r'.\endo_vasc_segmentation\weights\outer_maskrcnnweights.pth'  # path to your saved model
    outer_num_classes = 2  # background + endodermis
    outer_input_folder = os.path.join(input_folder, 'DAPI_cropped_normalized')  # folder containing images to predict on
    outer_output_folder = os.path.join(input_folder, 'outer_masks')  # folder to save

    model, device = load_model(outer_model_path, outer_num_classes)
    process_images(0.1, model, outer_input_folder, outer_output_folder, device)

    
if __name__ == '__main__':
    input_folder = r'C:\Users\Yifei\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\Kevin_Cropped_Images\All_Folders_Compiled_for_test_processed'
    predict_inner_outer_masks(input_folder)

