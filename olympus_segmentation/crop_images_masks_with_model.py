import os
import torch
import segmentation_models_pytorch as smp
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_val_transforms():
    return A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(),
        ToTensorV2(),
    ],
        additional_targets={
            "mask": "mask"
        })


class FlexibleDataset:
    """
    Flexible dataset that can handle any folder structure and file naming convention.
    """
    def __init__(self, root_dir, channels, transform=None, manual_annotation='True'):
        self.root_dir = root_dir
        self.channels = channels
        self.transform = transform
        self.manual_annotation = manual_annotation
        self.data = []
        
        print(f"Scanning directory structure: {root_dir}")
        self._discover_samples()
        
    def _discover_samples(self):

        if self.manual_annotation == 'True':
            self._discover_samples_with_annotations()
        else:
            self._discover_samples_without_annotations()
    
    def _discover_samples_with_annotations(self):
        """
        Find samples by looking for .txt annotation files and corresponding image folders.
        """
        # Find all .txt files recursively
        txt_files = list(Path(self.root_dir).rglob("*.txt"))
        print(f"Found {len(txt_files)} annotation files")
        
        for txt_file in txt_files:
            # Get the base name (without .txt extension)
            sample_base_name = txt_file.stem
            
            # Look for corresponding image folder
            potential_image_dir = txt_file.parent / sample_base_name
            
            if potential_image_dir.exists() and potential_image_dir.is_dir():
                print(f"Processing sample: {sample_base_name}")
                print(f"  Annotation: {txt_file}")
                print(f"  Images dir: {potential_image_dir}")
                
                # Try to load the image stack
                image_stack = self._load_image_stack(potential_image_dir, sample_base_name)
                if image_stack is None:
                    print(f"  Skipping {sample_base_name} - could not load images")
                    continue
                
                # Load annotation
                annotation = self._load_annotation(txt_file, image_stack)
                if annotation is None:
                    print(f"  Skipping {sample_base_name} - could not load annotation")
                    continue
                
                self.data.append((image_stack, annotation, sample_base_name, str(potential_image_dir), str(txt_file)))
                print(f"  Successfully loaded sample {sample_base_name}")
            else:
                print(f"  Warning: No image directory found for {sample_base_name}")
    
    def _discover_samples_without_annotations(self):
        """
        Find samples by looking for image folders directly (no annotations required).
        """
        print("Scanning for image folders (no annotations required)...")
        
        # Find all directories that contain image files
        for root, dirs, files in os.walk(self.root_dir):
            root_path = Path(root)
            
            # Check if this directory contains image files
            image_files = []
            image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.ome.tif']
            for ext in image_extensions:
                image_files.extend(list(root_path.glob(ext)))
            
            if len(image_files) >= len(self.channels):  # Must have at least as many files as channels
                # Use the directory name as sample ID
                sample_base_name = root_path.name
                
                print(f"Processing sample: {sample_base_name}")
                print(f"  Images dir: {root_path}")
                print(f"  Found {len(image_files)} image files")
                
                # Try to load the image stack
                image_stack = self._load_image_stack(root_path, sample_base_name)
                if image_stack is None:
                    print(f"  Skipping {sample_base_name} - could not load images")
                    continue
                
                # Create a dummy txt file path for consistency
                dummy_txt_path = root_path / f"{sample_base_name}.txt"
                
                self.data.append((image_stack, sample_base_name, str(root_path), str(dummy_txt_path)))
                print(f"  Successfully loaded sample {sample_base_name}")
        
        print(f"Total samples loaded: {len(self.data)}")
    
    def _find_channel_files(self, image_dir, sample_base_name):
        """
        Find channel files in the image directory using flexible matching.
        """
        image_files = {}
        
        # Get all image files in the directory
        image_extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg', '*.ome.tif']
        all_files = []
        for ext in image_extensions:
            all_files.extend(list(Path(image_dir).glob(ext)))
        
        print(f"    Found {len(all_files)} image files: {[f.name for f in all_files]}")
        
        # For each channel, find the best matching file
        for channel in self.channels:
            best_match = None
            for file_path in all_files:
                filename = file_path.name.upper()
                
                # Check if this file contains the channel name
                if channel.upper() in filename:
                    # Prefer files that also contain the sample name
                    if any(part.upper() in filename for part in sample_base_name.split('_')):
                        best_match = file_path
                        break
                    elif best_match is None:
                        best_match = file_path
            
            if best_match:
                image_files[channel] = best_match
                print(f"    {channel}: {best_match.name}")
            else:
                print(f"    WARNING: No file found for channel {channel}")
                return None
        
        return image_files
    
    def _load_image_stack(self, image_dir, sample_base_name):
        """
        Load images from all channels for a given sample.
        """
        # Find channel files
        channel_files = self._find_channel_files(image_dir, sample_base_name)
        if channel_files is None:
            return None
        
        # Load images
        imgs = []
        for channel in self.channels:
            if channel not in channel_files:
                print(f"    Missing channel {channel}")
                return None
            
            img_path = channel_files[channel]
            print(f"    Loading {channel} from: {img_path}")
            
            # Load image
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"    ERROR: Could not load {img_path}")
                return None
            
            # Convert to 8-bit if necessary (for consistency with original code)
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            elif img.dtype == np.float32 or img.dtype == np.float64:
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
            
            imgs.append(img)
            print(f"    Loaded {channel}: shape={img.shape}, dtype={img.dtype}")
        
        # Stack images
        result = np.stack(imgs, axis=-1)
        print(f"    Final stack shape: {result.shape}")
        return result
    
    def _load_annotation(self, txt_file, image):
        """
        Load YOLO annotation and convert to mask.
        """
        if image is None:
            return None
        
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            print(f"    Loading annotation with {len(lines)} lines")
            
            for line_num, line in enumerate(lines):
                tokens = line.strip().split()
                if len(tokens) < 3 or len(tokens) % 2 == 0:
                    continue
                
                cls = int(tokens[0])
                coords = list(map(float, tokens[1:]))
                
                # Convert normalized coordinates to pixel coordinates
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * w)
                    y = int(coords[i+1] * h)
                    points.append([x, y])
                
                poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                
                if cls == 1:  # Assuming class 1 is AERE
                    cv2.fillPoly(mask, [poly], color=1)
                    print(f"      Added polygon for class {cls}")
            
            print(f"    Created mask with {np.sum(mask)} pixels")
            return mask
            
        except Exception as e:
            print(f"    ERROR loading annotation: {e}")
            return None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.manual_annotation == 'True':
            image, mask, sample_id, image_dir, txt_file = self.data[idx]
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            return image, mask, sample_id, image_dir, txt_file
        else:
            image, sample_id, image_dir, txt_file = self.data[idx]
            
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            
            return image, sample_id, image_dir, txt_file

def find_square_bounding_box(mask, padding=10):
    """
    Find a square bounding box around the predicted mask.
    """
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        h, w = mask.shape
        center_y, center_x = h // 2, w // 2
        size = min(h, w) // 4
        return max(0, center_x - size//2), max(0, center_y - size//2), \
               min(w, center_x + size//2), min(h, center_y + size//2)
    
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    width = x_max - x_min
    height = y_max - y_min
    size = max(width, height) + 2 * padding
    
    half_size = size // 2
    x1 = max(0, center_x - half_size)
    y1 = max(0, center_y - half_size)
    x2 = min(mask.shape[1], center_x + half_size)
    y2 = min(mask.shape[0], center_y + half_size)
    
    actual_width = x2 - x1
    actual_height = y2 - y1
    
    if actual_width != actual_height:
        min_size = min(actual_width, actual_height)
        x2 = x1 + min_size
        y2 = y1 + min_size
    
    return x1, y1, x2, y2

def transform_yolo_annotations(txt_file, original_width, original_height, crop_x1, crop_y1, crop_x2, crop_y2):
    """
    Transform YOLO annotations to account for cropping.
    
    Args:
        txt_file: Path to original YOLO annotation file
        original_width, original_height: Original image dimensions
        crop_x1, crop_y1, crop_x2, crop_y2: Crop coordinates in original image
        
    Returns:
        list: Transformed YOLO annotation lines that fall within the crop
    """
    transformed_annotations = []
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    
    try:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        print(f"    Processing {len(lines)} annotation lines")
        
        for line_num, line in enumerate(lines):
            tokens = line.strip().split()
            if len(tokens) < 3 or len(tokens) % 2 == 0:
                continue
            
            class_id = int(tokens[0])
            coords = list(map(float, tokens[1:]))
            
            # Convert normalized coordinates to pixel coordinates in original image
            original_points = []
            for i in range(0, len(coords), 2):
                x_norm = coords[i]
                y_norm = coords[i+1]
                x_pixel = x_norm * original_width
                y_pixel = y_norm * original_height
                original_points.append((x_pixel, y_pixel))
            
            # Check if polygon intersects with crop region
            polygon_in_crop = []
            for x_pixel, y_pixel in original_points:
                # Transform to crop coordinates
                x_crop = x_pixel - crop_x1
                y_crop = y_pixel - crop_y1
                
                # Check if point is within crop bounds (with small tolerance)
                if -10 <= x_crop <= crop_width + 10 and -10 <= y_crop <= crop_height + 10:
                    # Clamp to crop boundaries
                    x_crop = max(0, min(crop_width, x_crop))
                    y_crop = max(0, min(crop_height, y_crop))
                    
                    # Normalize to new crop dimensions
                    x_norm_new = x_crop / crop_width
                    y_norm_new = y_crop / crop_height
                    
                    # Clamp normalized coordinates to [0, 1]
                    x_norm_new = max(0.0, min(1.0, x_norm_new))
                    y_norm_new = max(0.0, min(1.0, y_norm_new))
                    
                    polygon_in_crop.extend([x_norm_new, y_norm_new])
            
            # Only keep polygons that have at least 3 points (6 coordinates) in the crop
            if len(polygon_in_crop) >= 6:
                # Create new YOLO line
                new_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in polygon_in_crop])
                transformed_annotations.append(new_line)
                print(f"      Transformed polygon for class {class_id}: {len(polygon_in_crop)//2} points")
            else:
                print(f"      Skipped polygon for class {class_id}: only {len(polygon_in_crop)//2} points in crop")
        
        print(f"    Result: {len(transformed_annotations)} annotations kept after cropping")
        return transformed_annotations
        
    except Exception as e:
        print(f"    ERROR transforming annotations: {e}")
        return []

def save_yolo_annotations(annotation_lines, save_path):
    """
    Save YOLO annotation lines to a text file.
    
    Args:
        annotation_lines: List of YOLO annotation strings
        save_path: Path to save the annotation file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w') as f:
            for line in annotation_lines:
                f.write(line + '\n')
        print(f"    Saved {len(annotation_lines)} annotations to: {save_path}")
        return True
    except Exception as e:
        print(f"    ERROR saving annotations to {save_path}: {e}")
        return False

def process_recursive_cropping(model, input_root, output_root, channels, with_annotations=True):
    """
    Process all samples in a directory tree recursively and save cropped versions
    with preserved folder structure and original file names.
    
    Args:
        model: Trained segmentation model
        input_root: Root directory containing input data
        output_root: Root directory for output data
        channels: List of channel names to look for
        with_annotations: True if input has annotation files, False for inference-only
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create dataset
    manual_annotation = 'True' if with_annotations else 'False'
    dataset = FlexibleDataset(input_root, channels, transform=get_val_transforms(), manual_annotation=manual_annotation)
    
    if len(dataset) == 0:
        print("No samples found to process!")
        return []
    
    processed_samples = []
    
    for idx in range(len(dataset)):
        try:
            # Get data from dataset
            if dataset.manual_annotation == 'True':
                image, true_mask, sample_id, image_dir, txt_file = dataset[idx]
                has_annotations = True
            else:
                image, sample_id, image_dir, txt_file = dataset[idx]
                has_annotations = False
                true_mask = None
            
            print(f"Processing sample {idx + 1}/{len(dataset)}: {sample_id}")
            
            # Convert image to numpy if it's a tensor
            if torch.is_tensor(image):
                if image.dim() == 3 and image.shape[0] == len(channels):
                    image_np = image.permute(1, 2, 0).cpu().numpy()
                else:
                    image_np = image.cpu().numpy()
            else:
                image_np = image
            
            print(f"  Image shape: {image_np.shape}, dtype: {image_np.dtype}")
            print(f"  Image value range: [{image_np.min()}, {image_np.max()}]")
            
            # Predict with model
            image_tensor = image.unsqueeze(0).to(device) if not torch.is_tensor(image) else image.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(image_tensor)
            
            # Convert to binary mask and find bounding box
            probs = torch.sigmoid(logits)
            pred_mask = (probs > 0.5).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            x1, y1, x2, y2 = find_square_bounding_box(pred_mask, padding=50)  # Generous padding for safety
            
            print(f"  Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
            
            # Calculate relative path from input_root to preserve structure
            image_dir_path = Path(image_dir)
            txt_file_path = Path(txt_file)
            input_root_path = Path(input_root)
            
            # Get relative path from input root to the sample directory
            try:
                relative_path = image_dir_path.relative_to(input_root_path)
                relative_txt_path = txt_file_path.relative_to(input_root_path)
            except ValueError:
                # Fallback if paths are not relative
                relative_path = Path(sample_id)
                relative_txt_path = Path(sample_id + '.txt')
            
            # Create output directory preserving structure
            output_image_dir = Path(output_root) / relative_path
            output_txt_path = Path(output_root) / relative_txt_path
            
            os.makedirs(output_image_dir, exist_ok=True)
            os.makedirs(output_txt_path.parent, exist_ok=True)
            
            print(f"  Output dir: {output_image_dir}")
            
            # Process and save each channel with original filenames
            if has_annotations:
                channel_files = dataset._find_channel_files(image_dir_path, sample_id)
            else:
                # For inference-only mode, we need to get channel files differently
                channel_files = dataset._find_channel_files(image_dir_path, sample_id)
            
            for ch_idx, channel in enumerate(channels):
                original_channel = image_np[:, :, ch_idx]
                cropped_channel = original_channel[y1:y2, x1:x2]
                
                print(f"    Processing channel {channel}")
                print(f"    Original range: [{original_channel.min()}, {original_channel.max()}]")
                print(f"    Cropped range: [{cropped_channel.min()}, {cropped_channel.max()}]")
                
                # Handle data type conversion while preserving intensity
                if cropped_channel.max() == 0 and cropped_channel.min() == 0:
                    print(f"    WARNING: Channel {channel} is all zeros after cropping!")
                    cropped_channel_to_save = np.zeros_like(cropped_channel, dtype=np.uint16)
                else:
                    if cropped_channel.dtype == np.float32 or cropped_channel.dtype == np.float64:
                        min_val, max_val = cropped_channel.min(), cropped_channel.max()
                        
                        if max_val <= 1.0 and min_val >= 0.0:
                            cropped_channel_to_save = (cropped_channel * 65535).astype(np.uint16)
                            print(f"    Preserved [0,1] -> [0,65535] scaling (uint16)")
                        elif max_val <= 1.0 and min_val >= -1.0:
                            cropped_channel_to_save = ((cropped_channel + 1) * 32767.5).astype(np.uint16)
                            print(f"    Preserved [-1,1] -> [0,65535] scaling (uint16)")
                        else:
                            if max_val > min_val:
                                cropped_channel_normalized = (cropped_channel - min_val) / (max_val - min_val)
                                cropped_channel_to_save = (cropped_channel_normalized * 65535).astype(np.uint16)
                                print(f"    Preserved intensity scaling: [{min_val}, {max_val}] -> [0, 65535] (uint16)")
                            else:
                                cropped_channel_to_save = np.full_like(cropped_channel, 32767, dtype=np.uint16)
                    elif cropped_channel.dtype == np.uint8:
                        cropped_channel_to_save = (cropped_channel.astype(np.uint16) * 257)
                        print(f"    Scaled uint8 to uint16 for preservation")
                    elif cropped_channel.dtype == np.uint16:
                        cropped_channel_to_save = cropped_channel
                        print(f"    Already uint16, no conversion needed")
                    else:
                        # Handle other data types
                        min_val, max_val = float(cropped_channel.min()), float(cropped_channel.max())
                        if max_val > min_val:
                            cropped_channel_normalized = (cropped_channel.astype(np.float64) - min_val) / (max_val - min_val)
                            cropped_channel_to_save = (cropped_channel_normalized * 65535).astype(np.uint16)
                            print(f"    Preserved generic type scaling: [{min_val}, {max_val}] -> [0, 65535] (uint16)")
                        else:
                            cropped_channel_to_save = np.full_like(cropped_channel, 32767, dtype=np.uint16)
                
                # Save with original filename
                if channel in channel_files:
                    original_filename = channel_files[channel].name
                    output_path = output_image_dir / original_filename
                    
                    success = cv2.imwrite(str(output_path), cropped_channel_to_save)
                    if success:
                        print(f"    Saved: {original_filename}")
                    else:
                        print(f"    ERROR saving: {original_filename}")
                else:
                    print(f"    ERROR: No original file found for channel {channel}")
            
            # Transform original YOLO annotations to cropped coordinates (if annotations exist)
            if has_annotations:
                print(f"  Transforming YOLO annotations for crop region...")
                original_height, original_width = image_np.shape[:2]
                
                transformed_annotations = transform_yolo_annotations(
                    txt_file=txt_file,
                    original_width=original_width,
                    original_height=original_height,
                    crop_x1=x1,
                    crop_y1=y1,
                    crop_x2=x2,
                    crop_y2=y2
                )
                
                if transformed_annotations:
                    save_yolo_annotations(transformed_annotations, str(output_txt_path))
                    print(f"  Saved {len(transformed_annotations)} transformed annotations")
                else:
                    print(f"    No annotations fall within the crop region")
                    # Create empty annotation file
                    with open(str(output_txt_path), 'w') as f:
                        pass  # Empty file
                    print(f"    Created empty annotation file: {output_txt_path}")
            else:
                print(f"  No annotations to process (inference-only mode)")
            
            # Track metrics
            sample_info = {
                'sample_id': sample_id,
                'input_path': str(image_dir_path),
                'output_path': str(output_image_dir),
                'bounding_box': (x1, y1, x2, y2),
                'cropped_shape': cropped_channel_to_save.shape,
                'pred_mask_area': np.sum(pred_mask == 1),
                'cropped_pred_mask_area': np.sum(pred_mask[y1:y2, x1:x2] == 1),
            }
            
            if has_annotations:
                sample_info.update({
                    'original_annotations_count': len(open(txt_file, 'r').readlines()),
                    'transformed_annotations_count': len(transformed_annotations),
                })
            
            processed_samples.append(sample_info)
            print(f"  Successfully processed {sample_id}")
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return processed_samples

# --------------------- Main ---------------------
if __name__ == '__main__':
    # Set your directories
    channels = ['DAPI', 'FITC', 'TRITC']
    model_path = r'C:\Users\yifei\Box\Carney Lab Shared\Projects\ROOTS-ProjectFolder\yifei\olympus_segmentation\weights\whole_root_binary_model_trained_tamera.pth'
    input_root = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Projects/ROOTS-ProjectFolder/yifei/data_for_publication/For_labeling_service'
    output_root = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Projects/ROOTS-ProjectFolder/yifei/data_for_publication/For_labeling_service_cropped'
    
    # Load the model
    model = smp.Unet(
        encoder_name='resnet34',
        in_channels=3,
        classes=1,
        activation=None
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Using device: {device}")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    
    # Process all samples recursively (with annotations)
    processed_samples = process_recursive_cropping(
        model=model,
        input_root=input_root,
        output_root=output_root,
        channels=channels,
        with_annotations=True  # Set to False if your folders have no annotation files
    )
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed {len(processed_samples)} samples.")
    print(f"Results saved to: {output_root}")
