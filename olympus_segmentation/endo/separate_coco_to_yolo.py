#!/usr/bin/env python3
"""
Split a COCO-format JSON into per-image YOLO-seg .txt files.

Each line in the output .txt is:
  <yolo_class> x1_norm y1_norm x2_norm y2_norm ... xN_norm yN_norm

Assumes all segmentations are simple polygons (lists of [x,y,...]).
"""

import json
import os
from pathlib import Path

# === USER CONFIGURATION ===
COCO_JSON = r'C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\SQR & SRN\Lucas_167images_Outer_Endodermis_Annotations.json'
OUTPUT_DIR = r'C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\SQR & SRN\yolo_seg_labels_outer'
# Map COCO category_id → YOLO class index
CATEGORY_MAP = {
    1: 1,    # your "inner" category (id=1) becomes YOLO class 0
    # if you add more categories, do e.g.
    # 2: 1,
    # 3: 2,
}
# ==========================

def main():
    # load coco
    coco = json.load(open(COCO_JSON, 'r'))
    images = {img['id']: img for img in coco['images']}
    annotations = coco['annotations']

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # group annotations by image_id
    ann_by_image = {}
    for ann in annotations:
        img_id = ann['image_id']
        ann_by_image.setdefault(img_id, []).append(ann)

    for img_id, img_info in images.items():
        fname = img_info['file_name']
        w, h = img_info['width'], img_info['height']
        base = Path(fname).stem
        out_txt = Path(OUTPUT_DIR) / f"{base}.txt"

        lines = []
        for ann in ann_by_image.get(img_id, []):
            cat_id = ann['category_id']
            if cat_id not in CATEGORY_MAP:
                # skip any categories you don't want
                continue
            yolo_cls = CATEGORY_MAP[cat_id]

            # segmentation may be list of polygons
            for poly in ann.get('segmentation', []):
                # poly is flat [x1,y1,x2,y2,...]
                # normalize each coordinate
                norm = []
                for idx, coord in enumerate(poly):
                    if idx % 2 == 0:
                        norm.append(coord / w)
                    else:
                        norm.append(coord / h)
                # format line
                coords_str = " ".join(f"{c:.6f}" for c in norm)
                lines.append(f"{yolo_cls} {coords_str}")

        # write file (even if empty, to keep consistency)
        with open(out_txt, 'w') as f:
            f.write("\n".join(lines))

        print(f"→ Wrote {len(lines)} polygons to {out_txt}")

    print("All done.")

if __name__ == '__main__':
    main()
