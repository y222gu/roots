import os
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import tifffile
from pathlib import Path
from scipy.ndimage import gaussian_filter, median_filter

def find_channel_images(sample_dir):
    """
    Find all channel images in a sample directory.
    Returns a dictionary with channel names as keys.
    Handles various naming conventions including CH(NAME) format.
    Treats FITC and GFP as interchangeable (both map to 'FITC' key).
    """
    channels = {}
    
    # Get all .tif files in the directory
    tif_files = [f for f in os.listdir(sample_dir) 
                 if f.endswith(('.tif', '.tiff', '.ome.tif'))]
    
    print(f"  Found .tif files: {tif_files}")
    
    # Match files to channels - check in priority order to avoid conflicts
    # IMPORTANT: Check FITC/GFP BEFORE TRITC to avoid assigning FITC files to TRITC
    
    # DAPI channel
    for tif_file in tif_files:
        if 'CH(DAPI)' in tif_file or 'DAPI' in tif_file or tif_file.endswith('_CH1.ome.tif'):
            if 'DAPI' not in channels:
                channels['DAPI'] = os.path.join(sample_dir, tif_file)
                print(f"    Matched '{tif_file}' -> DAPI")
    
    # GFP/FITC channel (green fluorescence) - CHECK THIS BEFORE TRITC
    # Map both FITC and GFP to 'FITC' key for consistency
    for tif_file in tif_files:
        if 'CH(FITC)' in tif_file or 'CH(GFP)' in tif_file or 'FITC' in tif_file or 'GFP' in tif_file or tif_file.endswith('_CH2.ome.tif'):
            if 'FITC' not in channels:
                channels['FITC'] = os.path.join(sample_dir, tif_file)
                print(f"    Matched '{tif_file}' -> FITC (GFP-compatible)")
    
    # TRITC channel (red fluorescence) - only match explicit TRITC labels
    for tif_file in tif_files:
        if 'CH(TRITC)' in tif_file or 'TRITC' in tif_file or tif_file.endswith('_CH3.ome.tif'):
            if 'TRITC' not in channels:
                channels['TRITC'] = os.path.join(sample_dir, tif_file)
                print(f"    Matched '{tif_file}' -> TRITC")
    
    return channels

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
                          contrast_enhancement=1.2,
                          channel_name=None,
                          per_channel_contrast=None):
    """
    Fast preprocessing - removes redundant steps.
    
    Steps:
    1. Background subtraction (if enabled)
    2. Normalization to 0-255 with percentile clipping
    3. Smoothing (Gaussian)
    4. Contrast enhancement (per-channel or global)
    
    Args:
        img: Input image
        apply_background_subtraction: Whether to subtract background
        apply_smoothing: Apply smoothing
        smooth_sigma: Gaussian smoothing sigma
        contrast_enhancement: Default contrast multiplier (1.0 = no change)
        channel_name: Name of the channel being processed
        per_channel_contrast: Dictionary mapping channel names to contrast factors
    
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
    
    # Step 4: Contrast enhancement via PIL (per-channel or global)
    # Determine which contrast factor to use
    # Support both FITC and GFP keys in per_channel_contrast
    if per_channel_contrast and channel_name:
        # Try exact match first
        if channel_name in per_channel_contrast:
            contrast_factor = per_channel_contrast[channel_name]
        # If channel is FITC, also try GFP
        elif channel_name == 'FITC' and 'GFP' in per_channel_contrast:
            contrast_factor = per_channel_contrast['GFP']
        # If channel is GFP, also try FITC
        elif channel_name == 'GFP' and 'FITC' in per_channel_contrast:
            contrast_factor = per_channel_contrast['FITC']
        else:
            contrast_factor = contrast_enhancement
    else:
        contrast_factor = contrast_enhancement
    
    if contrast_factor != 1.0:
        start = time.time()
        pil = Image.fromarray(img8, mode='L')
        enhancer = ImageEnhance.Contrast(pil)
        pil = enhancer.enhance(contrast_factor)
        img8 = np.array(pil, dtype=np.uint8)
        print(f" [contrast: {time.time()-start:.2f}s ({contrast_factor}x)]", end="")
    
    total = time.time() - total_start
    print(f" [TOTAL: {total:.2f}s]")
    
    return img8

def create_composite_image(channels_dict, output_path, channel_colors=None, channels_to_use=None, 
                          preprocess_params=None, per_channel_contrast=None):
    """
    Create a colored composite image from selected channels and save as TIFF.
    Handles FITC/GFP interchangeably.
    
    Args:
        per_channel_contrast: Dictionary mapping channel names to contrast enhancement factors
                            e.g., {'DAPI': 1.1, 'FITC': 1.3, 'TRITC': 1.2}
                            Note: Both 'FITC' and 'GFP' keys are supported
    """
    # Default color mapping (supports both FITC and GFP)
    if channel_colors is None:
        channel_colors = {
            'DAPI': 'blue',
            'FITC': 'green',
            'GFP': 'green',  # Same as FITC
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
    # Handle both FITC and GFP in channels_to_use
    if channels_to_use is not None:
        # Normalize channels_to_use to support both FITC and GFP
        normalized_channels_to_use = set()
        for ch in channels_to_use:
            normalized_channels_to_use.add(ch)
            # If user specifies GFP, also accept FITC
            if ch == 'GFP':
                normalized_channels_to_use.add('FITC')
            # If user specifies FITC, also accept GFP
            elif ch == 'FITC':
                normalized_channels_to_use.add('GFP')
        
        channels_dict = {k: v for k, v in channels_dict.items() if k in normalized_channels_to_use}
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
            
            # Apply preprocessing with per-channel contrast
            proj_pre = preprocess_projection(
                proj,
                apply_background_subtraction=preprocess_params.get('apply_background_subtraction', True),
                apply_smoothing=preprocess_params.get('apply_smoothing', True),
                smooth_sigma=preprocess_params.get('smooth_sigma', 1.0),
                contrast_enhancement=preprocess_params.get('contrast_enhancement', 1.2),
                channel_name=channel_name,
                per_channel_contrast=per_channel_contrast
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
        # Support both FITC and GFP in color lookup
        color_name = channel_colors.get(channel_name)
        if color_name is None:
            # If exact match not found, try alternative
            if channel_name == 'FITC' and 'GFP' in channel_colors:
                color_name = channel_colors['GFP']
            elif channel_name == 'GFP' and 'FITC' in channel_colors:
                color_name = channel_colors['FITC']
            else:
                color_name = 'gray'
        
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


def process_directory(input_dir, output_dir, channel_colors=None, channels_to_use=None, 
                     preprocess_params=None, per_channel_contrast=None):
    """
    Recursively process all sample directories.
    
    Args:
        per_channel_contrast: Dictionary mapping channel names to contrast enhancement factors
                            Note: Both 'FITC' and 'GFP' keys are supported
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Walk through all directories
    for root, dirs, files in os.walk(input_path):
        # Check if this directory contains TIFF files
        tif_files = [f for f in files if f.endswith(('.tif', '.tiff', '.ome.tif', '*.ome.tiff'))]
        
        if tif_files:
            # Try to find channel images
            channels = find_channel_images(root)
            
            # If we found at least one channel, process this sample
            if channels:
                # Create relative path structure in output
                rel_path = Path(root).relative_to(input_path)
                sample_name = rel_path.parts[-1] if rel_path.parts else 'root'
                
                # Create descriptive output filename
                parent_parts = list(rel_path.parts[:-1]) if len(rel_path.parts) > 1 else []
                if parent_parts:
                    output_name = f"{'_'.join(parent_parts)}_{sample_name}_composite.tif"
                else:
                    output_name = f"{sample_name}_composite.tif"
                
                output_file = output_path / output_name
                
                print(f"\nProcessing: {rel_path}")
                print(f">>> Found {len(channels)} channels: {list(channels.keys())}")
                for ch_name, ch_path in channels.items():
                    print(f"    - {ch_name}: {os.path.basename(ch_path)}")
                
                # Create composite image with per-channel contrast
                create_composite_image(channels, str(output_file), 
                                     channel_colors=channel_colors,
                                     channels_to_use=channels_to_use,
                                     preprocess_params=preprocess_params,
                                     per_channel_contrast=per_channel_contrast)


if __name__ == "__main__":
    # Set your input and output directories
    input_directory = r'C:\Users\Yifei\Box\Carney Lab Shared\Projects\ROOTS-ProjectFolder\yifei\data_for_publication\temp_cropped'
    output_directory = r'C:\Users\Yifei\Box\Carney Lab Shared\Projects\ROOTS-ProjectFolder\yifei\data_for_publication\temp_cropped_colored'

    # Color assignment for all channels
    # Both FITC and GFP are supported and map to the same color
    channel_colors = {
        'DAPI': 'blue',
        'FITC': 'yellow',     # Works with both FITC and GFP files
        'GFP': 'yellow',      # Explicit GFP mapping (same as FITC)
        'TRITC': 'red'
    }
    
    # Can specify either FITC or GFP - both will work
    channels_to_use = ['DAPI', 'FITC']  # Will also accept GFP-named files
    
    # FAST preprocessing - only essential steps
    preprocess_params = {
        'apply_background_subtraction': False,
        'apply_smoothing': False,       # Set to False for SHARPER images, True for SMOOTHER
        'smooth_sigma': 1.0,            # Only used if apply_smoothing=True
        'contrast_enhancement': 1.0     # Default (will be overridden by per_channel_contrast)
    }

    # Define separate contrast enhancement factors for each channel
    # Both FITC and GFP keys are supported
    per_channel_contrast = {
        'DAPI': 1.5,      # Lower enhancement for DAPI
        'FITC': 1,      # Enhancement for FITC (also applies to GFP)
        'GFP': 1,       # Same as FITC
        'TRITC': 1.1      # Medium enhancement for TRITC
    }

    print(f"Starting processing...")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Channels to use: {channels_to_use}")
    print(f"Color assignments: {channel_colors}")
    print(f"Per-channel contrast: {per_channel_contrast}")
    print("="*60)
    
    process_directory(input_directory, output_directory, 
                     channel_colors=channel_colors,
                     channels_to_use=channels_to_use,
                     preprocess_params=preprocess_params,
                     per_channel_contrast=per_channel_contrast)
    
    print("\n" + "="*60)
    print("Processing complete!")