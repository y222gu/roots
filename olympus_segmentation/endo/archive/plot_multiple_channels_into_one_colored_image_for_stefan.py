import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import tifffile
from pathlib import Path
from scipy.ndimage import gaussian_filter, median_filter

def extract_sample_name(filename):
    """
    Extract sample name from filename.
    Handles format like: Experiment_9_Minimedusa_01_Experiment_9_Minimedusa_01_raw_CH(DAPI)_CH1.ome.tif
    Returns the unique identifier part before "_raw" or before the channel designation.
    """
    # Remove extension
    name = filename.replace('.ome.tif', '').replace('.tiff', '').replace('.tif', '')
    
    # Split on "_raw" to get the sample identifier
    if '_raw_' in name:
        sample_id = name.split('_raw_')[0]
    else:
        # Fallback: remove the channel info from the end
        for suffix in ['_CH(DAPI)_CH1', '_CH(FITC)_CH2', '_CH(TRITC)_CH3', 
                       '_CH1', '_CH2', '_CH3']:
            if suffix in name:
                sample_id = name.replace(suffix, '')
                break
        else:
            sample_id = name
    
    return sample_id


def group_files_by_sample(file_list):
    """
    Group channel files by their sample name.
    Returns a dictionary: {sample_name: [file1, file2, file3, ...]}
    """
    samples = {}
    
    for filename in file_list:
        if not filename.endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff')):
            continue
        
        sample_name = extract_sample_name(filename)
        
        if sample_name not in samples:
            samples[sample_name] = []
        
        samples[sample_name].append(filename)
    
    return samples


def load_and_project(image_path):
    """
    Load a TIFF stack and create maximum intensity Z-projection.
    Returns a 2D numpy array.
    """
    img = tifffile.imread(image_path)
    
    # If it's a 3D stack, do max projection along Z-axis
    if img.ndim == 3:
        projection = np.max(img, axis=0)
    else:
        projection = img
    
    return projection

def simple_background_subtraction(img, gaussian_sigma=30):
    """
    Fast background subtraction using Gaussian blur.
    Much faster than morphological operations on large images.
    
    Args:
        img: Input image (any dtype)
        gaussian_sigma: Sigma for Gaussian blur (larger = slower but better background removal)
    
    Returns:
        Background-subtracted image (preserves intensity, clips at 0)
    """
    import time
    start = time.time()
    
    img_f = img.astype(np.float64)
    
    # Use Gaussian blur to estimate smooth background
    # MUCH faster than morphological operations
    background = gaussian_filter(img_f, sigma=gaussian_sigma)
    
    # Subtract background but preserve signal
    result = img_f - background
    result = np.clip(result, 0, None)  # Don't allow negative values
    
    elapsed = time.time() - start
    print(f" [bg_sub: {elapsed:.2f}s]", end="")
    
    return result

def preprocess_projection(img,
                          apply_background_subtraction=True,
                          apply_smoothing=True,
                          smooth_sigma=1.0,
                          contrast_enhancement=1.2):
    """
    Fast preprocessing - removes redundant steps.
    
    Steps:
    1. Background subtraction (if enabled)
    2. Normalization to 0-255 with percentile clipping
    3. Smoothing (Gaussian)
    4. Contrast enhancement
    
    Args:
        img: Input image
        apply_background_subtraction: Whether to subtract background
        apply_smoothing: Apply smoothing
        smooth_sigma: Gaussian smoothing sigma
        contrast_enhancement: Contrast multiplier (1.0 = no change)
    
    Returns:
        Processed uint8 image
    """
    import time
    total_start = time.time()
    
    img_f = img.astype(np.float64)
    
    # Step 1: Background subtraction (uses Gaussian blur - very fast)
    if apply_background_subtraction:
        img_f = simple_background_subtraction(img_f, gaussian_sigma=30)
    
    # Step 2: Normalize to 0-255 using percentile clipping
    start = time.time()
    p_low = np.percentile(img_f, 0.5)
    p_high = np.percentile(img_f, 99.5)
    
    if p_high > p_low:
        img_f = np.clip(img_f, p_low, p_high)
        img8 = ((img_f - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
    else:
        img8 = np.zeros_like(img_f, dtype=np.uint8)
    print(f" [norm: {time.time()-start:.2f}s]", end="")
    
    # Step 3: Gaussian smoothing
    if apply_smoothing and smooth_sigma > 0:
        start = time.time()
        img8_f = img8.astype(np.float32)
        img8_f = gaussian_filter(img8_f, sigma=smooth_sigma)
        img8 = np.clip(img8_f, 0, 255).astype(np.uint8)
        print(f" [smooth: {time.time()-start:.2f}s]", end="")
    
    # Step 4: Contrast enhancement via PIL
    if contrast_enhancement != 1.0:
        start = time.time()
        pil = Image.fromarray(img8, mode='L')
        enhancer = ImageEnhance.Contrast(pil)
        pil = enhancer.enhance(contrast_enhancement)
        img8 = np.array(pil, dtype=np.uint8)
        print(f" [contrast: {time.time()-start:.2f}s]", end="")
    
    total = time.time() - total_start
    print(f" [TOTAL: {total:.2f}s]")
    
    return img8

def create_composite_image(channels_dict, output_path, channel_colors=None, channels_to_use=None, preprocess_params=None):
    """
    Create a colored composite image from selected channels and save as TIFF.
    """
    # Default color mapping
    if channel_colors is None:
        channel_colors = {
            'DAPI': 'blue',
            'GFP': 'green',
            'TRITC': 'red'
        }

    # Default preprocess params - FAST and SIMPLE
    if preprocess_params is None:
        preprocess_params = {
            'apply_background_subtraction': True,
            'apply_smoothing': True,
            'smooth_sigma': 1.0,
            'contrast_enhancement': 1.2
        }

    # Filter channels if specific ones are requested
    if channels_to_use is not None:
        channels_dict = {k: v for k, v in channels_dict.items() if k in channels_to_use}
        print(f"  Using only channels: {list(channels_dict.keys())}")

    # Define color to RGB channel mapping
    color_to_rgb = {
        'red': [1, 0, 0],
        'green': [0, 1, 0],
        'blue': [0, 0, 1],
        'cyan': [0, 1, 1],
        'magenta': [1, 0, 1],
        'yellow': [1, 1, 0],
        'gray': [1, 1, 1],
        'white': [1, 1, 1]
    }

    # Load, project, and preprocess each channel
    projections = {}
    for channel_name, path in channels_dict.items():
        import time
        ch_start = time.time()
        print(f"\n  >>> Processing {channel_name}: {os.path.basename(path)}", end="")
        try:
            proj = load_and_project(path)  # raw projection
            print(f" (loaded in {time.time()-ch_start:.2f}s)", end="")
            print(f"\n      Raw: min={proj.min()}, max={proj.max()}, mean={proj.mean():.1f}, dtype={proj.dtype}", end="")
            
            # Apply preprocessing
            proj_pre = preprocess_projection(
                proj,
                apply_background_subtraction=preprocess_params.get('apply_background_subtraction', True),
                apply_smoothing=preprocess_params.get('apply_smoothing', True),
                smooth_sigma=preprocess_params.get('smooth_sigma', 1.0),
                contrast_enhancement=preprocess_params.get('contrast_enhancement', 1.2)
            )
            
            print(f"\n      After: min={proj_pre.min()}, max={proj_pre.max()}, mean={proj_pre.mean():.1f}, dtype={proj_pre.dtype}")
            print(f"      >>> Total time: {time.time()-ch_start:.2f}s")
            projections[channel_name] = proj_pre
        except Exception as e:
            print(f"\n      ERROR processing {channel_name}: {e}")
            import traceback
            traceback.print_exc()

    # Ensure all channels have the same dimensions
    if projections:
        shapes = [proj.shape for proj in projections.values()]
        if len(set(shapes)) > 1:
            print(f"  Warning: Channel dimensions don't match: {shapes}")
            min_h = min(s[0] for s in shapes)
            min_w = min(s[1] for s in shapes)
            for key in projections:
                projections[key] = projections[key][:min_h, :min_w]

    # Get image dimensions
    if not projections:
        print(f"  Error: No channels found")
        return False

    height, width = next(iter(projections.values())).shape

    # Normalize each channel individually to full 0-255 range before combining
    projections_norm = {}
    for channel_name, img in projections.items():
        img_f = img.astype(np.float32)
        max_val = img_f.max()
        min_val = img_f.min()
        
        if max_val > min_val:
            # Stretch to full range
            img_scaled = ((img_f - min_val) / (max_val - min_val)) * 255.0
        else:
            img_scaled = np.zeros_like(img_f)
        
        projections_norm[channel_name] = img_scaled
        print(f"  Channel {channel_name}: min={min_val}, max={max_val}, scaled_mean={img_scaled.mean():.1f}")

    # Create RGB composite image
    rgb_image = np.zeros((height, width, 3), dtype=np.float32)

    # Assign each channel to its corresponding color
    for channel_name in projections_norm:
        color_name = channel_colors.get(channel_name, 'gray')
        color_multipliers = color_to_rgb.get(color_name, [1, 1, 1])

        # Add this channel's contribution to RGB
        for i, multiplier in enumerate(color_multipliers):
            if multiplier > 0:
                rgb_image[:, :, i] += projections_norm[channel_name] * multiplier

        print(f"  Added {channel_name} ({color_name})")

    # Normalize each channel independently to 0-255
    # This preserves color saturation and prevents weak channels from being crushed
    for i in range(3):
        ch_max = rgb_image[:, :, i].max()
        if ch_max > 0:
            rgb_image[:, :, i] = (rgb_image[:, :, i] / ch_max * 255.0)
    
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    # Save as RGB TIFF
    tifffile.imwrite(output_path, rgb_image, photometric='rgb')
    print(f"  Saved colored composite: {output_path}")
    return True


def process_directory(input_dir, output_dir, channel_colors=None, channels_to_use=None, preprocess_params=None):
    """
    Process all samples in a flat directory structure.
    Groups files by sample name and creates composites.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all TIFF files in the input directory
    all_files = [f for f in os.listdir(input_path) 
                 if f.endswith(('.tif', '.tiff', '.ome.tif', '.ome.tiff'))]
    
    if not all_files:
        print(f"No TIFF files found in {input_dir}")
        return
    
    # Group files by sample name
    samples = group_files_by_sample(all_files)
    
    print(f"Found {len(samples)} unique samples")
    print("="*60)
    
    # Process each sample
    for sample_name, files in sorted(samples.items()):
        print(f"\nProcessing sample: {sample_name}")
        print(f"  Files in this sample: {len(files)}")
        for f in files:
            print(f"    - {f}")
        
        # Find channels in these files
        channels = {}
        
        # Create a temporary mapping for find_channel_images
        # We'll create a dict with full paths
        for filename in files:
            full_path = os.path.join(input_path, filename)
            
            # Match to channels
            if 'CH(DAPI)' in filename or '_CH1.ome.tif' in filename:
                if 'DAPI' not in channels:
                    channels['DAPI'] = full_path
                    print(f"    Matched '{filename}' -> DAPI")
            elif 'CH(TRITC)' in filename or '_CH3.ome.tif' in filename:
                if 'TRITC' not in channels:
                    channels['TRITC'] = full_path
                    print(f"    Matched '{filename}' -> TRITC")
            elif 'CH(FITC)' in filename or 'CH(GFP)' in filename or '_CH2.ome.tif' in filename:
                if 'GFP' not in channels:
                    channels['GFP'] = full_path
                    print(f"    Matched '{filename}' -> GFP")
        
        # If we found channels, create composite
        if channels:
            output_name = f"Sorghum_C10_{sample_name}_composite.tif"
            output_file = output_path / output_name
            
            print(f">>> Found {len(channels)} channels: {list(channels.keys())}")
            
            # Create composite image
            create_composite_image(channels, str(output_file), 
                                 channel_colors=channel_colors,
                                 channels_to_use=channels_to_use,
                                 preprocess_params=preprocess_params)
        else:
            print(f"  Warning: No channels found for sample {sample_name}")


if __name__ == "__main__":
    # Set your input and output directories
    input_directory = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Projects/ROOTS-ProjectFolder/Images_For_Labeller/Stefan'
    output_directory = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Projects/ROOTS-ProjectFolder/yifei/data_for_publication/For_labeling_service_sorghum_colored_all_three_colors'
    
    # Color assignment for all channels
    channel_colors = {
        'DAPI': 'blue',
        'GFP': 'yellow',
        'TRITC': 'magenta',
    }
    
    # Specify which channels to use
    channels_to_use = ['DAPI', 'GFP', 'TRITC']  # Only use DAPI, GFP, and TRITC channels
    
    # channels_to_use = None  # Uncomment to use ALL detected channels
    
    # FAST preprocessing - only essential steps
    preprocess_params = {
        'apply_background_subtraction': False,
        'apply_smoothing': False,       # Set to False for SHARPER images, True for SMOOTHER
        'smooth_sigma': 1.0,            # Only used if apply_smoothing=True
        'contrast_enhancement': 1.3     # Boost contrast for sharper appearance
    }

    print(f"Starting processing...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Channels to use: {channels_to_use}")
    print(f"Color assignments: {channel_colors}")
    print("="*60)
    
    process_directory(input_directory, output_directory, 
                     channel_colors=channel_colors,
                     channels_to_use=channels_to_use,
                     preprocess_params=preprocess_params)
    
    print("\n" + "="*60)
    print("Processing complete!")