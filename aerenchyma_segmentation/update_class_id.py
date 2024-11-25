from ultralytics.data.converter import convert_coco
import os

path = os.path.join(os.getcwd(), 'aerenchyma_segmentation','data', 'annotations')
convert_coco(labels_dir="", use_segments=True)
