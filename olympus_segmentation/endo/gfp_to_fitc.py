#!/usr/bin/env python3
import os
import sys

def rename_gfp_to_fitc(root_dir):
    # Walk the directory recursively
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            # Only target TIFF‐style files that contain “GFP” in the basename
            if 'BF' in fname and fname.lower().endswith(('.tif', '.tiff', '.ome.tif')):
                new_fname = fname.replace('BF', 'TRITC')
                old_path = os.path.join(dirpath, fname)
                new_path = os.path.join(dirpath, new_fname)

                if os.path.exists(new_path):
                    print(f"[!] skipping, target exists: {new_path}")
                else:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path!r} → {new_path!r}")

if __name__ == '__main__':
    # Default to your folder, or override via command-line
    DEFAULT_ROOT = r"C:\Users\Yifei\Documents\data_for_publication\test"
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = DEFAULT_ROOT
        print(f"No path given; defaulting to:\n  {root!r}")

    if not os.path.isdir(root):
        print(f"Error: “{root}” is not a directory.")
        sys.exit(1)

    rename_gfp_to_fitc(root)
