import os
import shutil

def move_files(source_dir, destination_dir):
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)
    
    # Walk through all subdirectories in source_dir
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                shutil.move(file_path, destination_dir)
                print(f"Moved: {file_path} -> {destination_dir}")
            except Exception as e:
                print(f"Error moving {file_path}: {e}")

# Example usage:
source_folder = r"C:\Users\Yifei\Documents\roots\tomato_segmentation\data\train\Folder4_image"
destination_folder = r"C:\Users\Yifei\Documents\roots\tomato_segmentation\data\train\Folder4_image_moved"

move_files(source_folder, destination_folder)
