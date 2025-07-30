import os

def remove_string_from_filenames(folder_path, string_to_remove):
    for filename in os.listdir(folder_path):
        if string_to_remove in filename:
            new_filename = filename.replace(string_to_remove, "")
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {filename} -> {new_filename}")

if __name__ == "__main__":
    folder_path = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\train\image'
    string_to_remove = '_image'
    
    if os.path.isdir(folder_path):
        remove_string_from_filenames(folder_path, string_to_remove)
    else:
        print("Invalid folder path.")