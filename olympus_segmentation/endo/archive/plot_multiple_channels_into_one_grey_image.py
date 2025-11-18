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
    """
    channels = {}
    
    # Get all .tif files in the directory
    tif_files = [f for f in os.listdir(sample_dir) 
                 if f.endswith(('.tif', '.tiff', '.ome.tif'))]
    
    print(f"  Found .tif files: {tif_files}")
    
    # Match files to channels - check in priority order to avoid conflicts
    # Priority: More specific patterns first (CH(DAPI)) before generic ones (CH1)
    
    # DAPI channel
    for tif_file in tif_files:
        if 'CH(DAPI)' in tif_file or 'DAPI' in tif_file or tif_file.endswith('_CH1.ome.tif'):
            if 'DAPI' not in channels:
                channels['DAPI'] = os.path.join(sample_dir, tif_file)
                print(f"    Matched '{tif_file}' -> DAPI")
    
    # TRITC channel
    for tif_file in tif_files:
        if 'CH(TRITC)' in tif_file or 'TRITC' in tif_file or tif_file.endswith('_CH2.ome.tif'):
            if 'TRITC' not in channels:
                channels['TRITC'] = os.path.join(sample_dir, tif_file)
                print(f"    Matched '{tif_file}' -> TRITC")
    
    # GFP channel (FITC is also green fluorescence)
    for tif_file in tif_files:
        if 'CH(FITC)' in tif_file or 'CH(GFP)' in tif_file or 'FITC' in tif_file or 'GFP' in tif_file or tif_file.endswith('_CH3.ome.tif'):
            if 'GFP' not in channels:
                channels['GFP'] = os.path.join(sample_dir, tif_file)
                print(f"    Matched '{tif_file}' -> GFP")
    
    return channels

def load_and_project(image_path):
    """
    Load a TIFF stack and create mean intensity Z-projection.
    Returns a 2D numpy array.
    """
    img = tifffile.imread(image_path)
    
    # If it's a 3D stack, do mean projection along Z-axis
    if img.ndim == 3:
        projection = np.mean(img, axis=0)
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


def create_composite_image(channels_dict, output_path, channels_to_use=None, preprocess_params=None):
    """
    Create a grayscale image from a single selected channel using mean intensity projection.
    Applies contrast enhancement AFTER the projection.
    """
    # Default preprocess params
    if preprocess_params is None:
        preprocess_params = {
            'apply_background_subtraction': True,
            'apply_smoothing': False,
            'smooth_sigma': 1.0,
            'contrast_enhancement': 1.3
        }

    # Filter channels if specific ones are requested
    if channels_to_use is not None:
        channels_dict = {k: v for k, v in channels_dict.items() if k in channels_to_use}
        print(f"  Using only channels: {list(channels_dict.keys())}")

    # Use the first selected channel
    if not channels_dict:
        print(f"  Error: No channels found")
        return False
    
    # Get the first channel from the filtered list
    channel_name = list(channels_dict.keys())[0]
    path = channels_dict[channel_name]
    
    print(f"\n  >>> Processing {channel_name}: {os.path.basename(path)}")
    
    try:
        # Step 1: Load and project (MEAN intensity projection)
        import time
        ch_start = time.time()
        proj = load_and_project(path)
        print(f"      Loaded in {time.time()-ch_start:.2f}s")
        print(f"      Raw projection - min={proj.min()}, max={proj.max()}, mean={proj.mean():.1f}, dtype={proj.dtype}")
        
        # Step 2: Background subtraction
        if preprocess_params.get('apply_background_subtraction', True):
            proj = simple_background_subtraction(proj, gaussian_sigma=30)
        
        # Step 3: Normalize to 0-255
        proj_f = proj.astype(np.float64)
        p_low = np.percentile(proj_f, 0.5)
        p_high = np.percentile(proj_f, 99.5)
        
        if p_high > p_low:
            proj_f = np.clip(proj_f, p_low, p_high)
            img8 = ((proj_f - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
        else:
            img8 = np.zeros_like(proj_f, dtype=np.uint8)
        
        print(f"      After normalization - min={img8.min()}, max={img8.max()}, mean={img8.mean():.1f}")
        
        # Step 4: Smoothing (if enabled)
        if preprocess_params.get('apply_smoothing', False):
            img8_f = img8.astype(np.float32)
            img8_f = gaussian_filter(img8_f, sigma=preprocess_params.get('smooth_sigma', 1.0))
            img8 = np.clip(img8_f, 0, 255).astype(np.uint8)
            print(f"      After smoothing - min={img8.min()}, max={img8.max()}, mean={img8.mean():.1f}")
        
        # Step 5: Contrast enhancement (AFTER projection)
        contrast_factor = preprocess_params.get('contrast_enhancement', 1.3)
        if contrast_factor != 1.0:
            pil = Image.fromarray(img8, mode='L')
            enhancer = ImageEnhance.Contrast(pil)
            pil = enhancer.enhance(contrast_factor)
            img8 = np.array(pil, dtype=np.uint8)
            print(f"      After contrast enhancement ({contrast_factor}x) - min={img8.min()}, max={img8.max()}, mean={img8.mean():.1f}")
        
        # Save as grayscale TIFF
        tifffile.imwrite(output_path, img8, photometric='minisblack')
        print(f"  >>> Saved grayscale image: {output_path}")
        print(f"  >>> Total time: {time.time()-ch_start:.2f}s")
        return True
        
    except Exception as e:
        print(f"  ERROR processing {channel_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(input_dir, output_dir, channels_to_use=None, preprocess_params=None):
    """
    Recursively process all sample directories.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Walk through all directories
    for root, dirs, files in os.walk(input_path):
        # Check if this directory contains TIFF files
        tif_files = [f for f in files if f.endswith(('.tif', '.tiff', '.ome.tif'))]
        
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
                
                # Create composite image
                create_composite_image(channels, str(output_file), 
                                     channels_to_use=channels_to_use,
                                     preprocess_params=preprocess_params)


if __name__ == "__main__":
    # Set your input and output directories
    input_directory = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Projects/ROOTS-ProjectFolder/yifei/data_for_publication/cropped_roots'
    output_directory = r'/Users/yifeigu/Library/CloudStorage/Box-Box/Carney Lab Shared/Projects/ROOTS-ProjectFolder/yifei/data_for_publication/cropped_roots_grey'
    
    # Select which channels to include in grayscale composite
    # channels_to_use = None  # Set to None to use ALL detected channels
    channels_to_use = ['DAPI']  # Only use DAPI and TRITC channels
    
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
    print("="*60)
    
    process_directory(input_directory, output_directory, 
                     channels_to_use=channels_to_use,
                     preprocess_params=preprocess_params)
    
    print("\n" + "="*60)
    print("Processing complete!")