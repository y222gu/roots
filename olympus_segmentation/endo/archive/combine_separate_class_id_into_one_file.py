#!/usr/bin/env python3
"""
Combine YOLO-seg inner (class 0) and outer (class 1) annotations
into single per-image label files.
"""

import os
import glob

# === USER CONFIGURATION ===
# Paths to your separate annotation folders:
INNER_DIR   = r'C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\SQR & SRN\LABELS\yolo_seg_labels_inner'
OUTER_DIR   = r'C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\SQR & SRN\LABELS\yolo_seg_labels_outer'

# Where to write the merged annotations:
OUTPUT_DIR  = r'C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\SQR & SRN\yolo_seg_labels_combined'

# Class IDs for each folder:
INNER_CLASS = 0
OUTER_CLASS = 1
# ===========================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# gather all unique stems from both folders
inner_txt = glob.glob(os.path.join(INNER_DIR, '*.txt'))
outer_txt = glob.glob(os.path.join(OUTER_DIR, '*.txt'))
stems = {
    os.path.splitext(os.path.basename(p))[0]
    for p in inner_txt + outer_txt
}

for stem in sorted(stems):
    merged_lines = []

    # load inner (0) annotations, if present
    inner_path = os.path.join(INNER_DIR, f"{stem}.txt")
    if os.path.isfile(inner_path):
        with open(inner_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts: 
                    continue
                coords = parts[1:]
                merged_lines.append(f"{INNER_CLASS} " + " ".join(coords))

    # load outer (1) annotations, if present
    outer_path = os.path.join(OUTER_DIR, f"{stem}.txt")
    if os.path.isfile(outer_path):
        with open(outer_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                coords = parts[1:]
                merged_lines.append(f"{OUTER_CLASS} " + " ".join(coords))

    # write merged file
    out_path = os.path.join(OUTPUT_DIR, f"{stem}.txt")
    with open(out_path, 'w') as out_f:
        out_f.write("\n".join(merged_lines))
        if merged_lines:
            out_f.write("\n")

    print(f"Wrote {len(merged_lines)} polygons → {out_path}")

print("✅ All files merged.")
