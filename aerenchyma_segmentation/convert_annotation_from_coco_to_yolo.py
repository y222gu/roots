import json
import os
with open(os.path.join(os.getcwd(), "aerenchyma_segmentation","instances_default.json")) as f:
    coco_data = json.load(f)

images = {img['id']: img for img in coco_data['images']}
categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}

for ann in coco_data['annotations']:
    img = images[ann['image_id']]
    width, height = img['width'], img['height']
    
    x, y, w, h = ann['bbox']
    x_center = (x + w / 2) / width
    y_center = (y + h / 2) / height
    w /= width
    h /= height
    
    class_id = categories[ann['category_id']]
    output_line = f"{class_id} {x_center} {y_center} {w} {h}\n"
    
    file_name = img['file_name'].split('.')[0] + ".txt"
    with open(file_name, 'a') as out_file:
        out_file.write(output_line)
