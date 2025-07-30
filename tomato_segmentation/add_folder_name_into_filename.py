import os

def add_folder_prefix(root_dir):
    # Walk through the directory tree
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            old_path = os.path.join(root, file)
            # Get the name of the immediate parent folder
            parent_folder = os.path.basename(root)
            
            # Only rename if the file does not already start with the folder name
            if not file.startswith(parent_folder + "_"):
                new_filename = f"{parent_folder}_{file}"
                new_path = os.path.join(root, new_filename)
                
                # Check to prevent overwriting an existing file
                if os.path.exists(new_path):
                    print(f"Skipping rename for {old_path}: {new_path} already exists.")
                else:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")

# Example usage:
root_directory = r"C:\Users\Yifei\Documents\roots\tomato_segmentation\data\train"
add_folder_prefix(root_directory)
