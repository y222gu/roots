import os
import numpy as np
import tifffile as tiff
import shutil
from endo_dataset import MultiChannelSegDataset


def save_preprocessed_images(
    data_dir: str,
    channels: list[str],
    out_base_dir: str,
    manual_annotation: bool = False
):
    """
    Walk `data_dir`, load each sample’s image stack, run `preprocess()`,
    save each channel back as float32 TIFF, and copy the sample's
    annotation file (.ome.txt) to the corresponding output folder,
    mirroring the original directory structure.
    """
    # instantiate dataset without augmentations
    ds = MultiChannelSegDataset(
        data_dir,
        channels,
        transform=None,
        manual_annotation=manual_annotation
    )

    for img_dir, ann_path, sample_id in ds.samples:
        # compute relative path of this sample's image folder
        rel_img = os.path.relpath(img_dir, data_dir)
        target_img_dir = os.path.join(out_base_dir, rel_img)
        os.makedirs(target_img_dir, exist_ok=True)

        # copy annotation file if it exists
        if ann_path:
            # get its relative directory under data_dir
            rel_ann_dir = os.path.relpath(os.path.dirname(ann_path), data_dir)
            target_ann_dir = os.path.join(out_base_dir, rel_ann_dir)
            os.makedirs(target_ann_dir, exist_ok=True)
            shutil.copy2(
                ann_path,
                os.path.join(target_ann_dir, os.path.basename(ann_path))
            )

        # load and preprocess the raw image stack
        raw_stack  = ds._load_image_stack(img_dir)      # H×W×C float32
        proc_stack = ds.preprocess(raw_stack)           # H×W×C float32 in [0,1]

        # write out each channel TIFF with original filename
        for ci, ch in enumerate(channels):
            orig_fn = next(
                f for f in os.listdir(img_dir)
                if f.endswith(('.ome.tif', '.tif', '.tiff')) and ch in f
            )
            out_path = os.path.join(target_img_dir, orig_fn)
            tiff.imwrite(out_path, proc_stack[:, :, ci].astype(np.float32))

    print(f"Done! Preprocessed images and annotations saved under {out_base_dir}")


if __name__ == '__main__':
    # adjust paths to your environment
    DATA_DIR   = r'C:\Users\Yifei\Documents\data_for_publication\test'
    OUTPUT_DIR = r'C:\Users\Yifei\Documents\data_for_publication\test_preprocessed'
    CHANNELS   = ['DAPI','FITC','TRITC']

    save_preprocessed_images(DATA_DIR, CHANNELS, OUTPUT_DIR)
