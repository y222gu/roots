import cv2
import os
import glob
import matplotlib.pyplot as plt

# 1) — adjust these paths if needed —
IMAGE_PATH      = r'C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\Images_Stefan\A2 20X PL FL Phase_ZProj_1_001_DAPI.tif'
ANNOTATION_PATH = r'C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\A2_DAPI_stefan.txt'
def plot_annotation(image_path, annotation_path):
    import matplotlib.pyplot as plt

    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image at {image_path!r}")
    h, w = img.shape[:2]

    # Define your class→name mapping
    class_names = {
        0: 'outer contour (epidermis/cortex boundary)',
        1: 'endodermis (outer border of the stele)',
        2: 'stele boundary (inner border)',
        3: 'aerenchyma air spaces / vascular‐bundle fragments',
    }

    # Colours (RGB tuples) for each class
    colors = {
        0: (1, 0, 0),   # red
        1: (0, 1, 0),   # green
        2: (0, 0, 1),   # blue
        3: (1, 1, 0),   # yellow
    }

    # Plot image and annotations
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax = plt.gca()

    found = set()
    with open(annotation_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            cls = int(parts[0])
            found.add(cls)
            # parse normalized coords → pixels
            coords = list(map(float, parts[1:]))
            pts = [(coords[i] * w, coords[i+1] * h) for i in range(0, len(coords), 2)]
            poly = plt.Polygon(pts,
                               fill=False,
                               edgecolor=colors.get(cls, (1, 1, 1)),
                               linewidth=1.2,
                               label=f'class {cls}')
            ax.add_patch(poly)

    plt.axis('off')
    plt.title('YOLO-Seg Annotations Overlay')
    plt.show()

    # Print out what labels we saw
    print("Classes found in the annotation file:", sorted(found))
    for cls in sorted(found):
        print(f"  • {cls} → {class_names.get(cls, 'Unknown')}")


# Define directories containing images and annotations.
# Update these paths according to your folder structure.
IMAGE_DIR = r"C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\Images_Stefan"
ANNOTATION_DIR = r"C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum"

# Loop through all .tif images in the image directory.
for image_path in glob.glob(os.path.join(IMAGE_DIR, "*.tif")):
    # Assume the annotation file has the same base name with a .txt extension.
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    annotation_path = os.path.join(ANNOTATION_DIR, base_name + ".txt")
    
    print(f"Processing image: {image_path}")
    
    try:
        plot_annotation(image_path, annotation_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
