import os
import shutil

def reorganize_zeiss_folder(root_dir):
    # 1) Move all .txt annotations into Zeiss/Annotations
    annotations_dir = os.path.join(root_dir, "Annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    for fname in os.listdir(root_dir):
        if fname.lower().endswith(".txt"):
            src = os.path.join(root_dir, fname)
            dst = os.path.join(annotations_dir, fname)
            shutil.move(src, dst)

    # 2) Group .tif images by sample into individual folders
    images_dir = os.path.join(root_dir, "3-Channel_Images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Could not find images folder: {images_dir}")

    for fname in os.listdir(images_dir):
        if not fname.lower().endswith(".tif"):
            continue
        # sample_name is the part before the first underscore, e.g. "Image 36"
        sample_name = fname.split("_", 1)[0]
        sample_dir = os.path.join(root_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)

        src = os.path.join(images_dir, fname)
        dst = os.path.join(sample_dir, fname)
        shutil.move(src, dst)

    # 3) (Optional) remove empty images_dir
    try:
        os.rmdir(images_dir)
    except OSError:
        # not empty or failed; you can remove manually if you like
        pass

if __name__ == "__main__":
    # <-- set this to the path of your Zeiss folder:
    root_dir = r"C:\Users\Yifei\Documents\data_for_publication\test\Zeiss"
    reorganize_zeiss_folder(root_dir)