import os
import shutil

def sort_and_clean_filenames(input_folder):

    # Create the output folder if it doesn't exist
    output_folder = os.path.join(input_folder +"_processed")
    os.makedirs(output_folder, exist_ok=True)

    # Create output folders for each stain
    channel_folders = ['DAPI', 'GFP', 'TRITC']
    for channel in channel_folders:
        os.makedirs(os.path.join(output_folder, channel), exist_ok=True)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        # Process only image files
        if filename.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png', '.txt')):
            print(f"Processing {filename}")

            # Extract image name and keep the name as A1, A2 ...
            sample_name = filename.split('_')[0]
            if "ROI" in sample_name:
                sample_name = sample_name.split("ROI")[0]

            # for Kevin's images
            # sample_name_1 = filename.split('_')[0]
            # if "ROI" in sample_name_1:
            #     sample_name_1 = sample_name_1.split("ROI")[0]
            # sample_name_2 = filename.split('_')[1]
            # sample_name_3 = filename.split(']')[1]
            # sample_name = sample_name_1 + "_" + sample_name_2 + "_" + sample_name_3

            # Determine the stain from the filename
            channel = next((s for s in channel_folders if s in filename), None)
            if not channel:
                print(f"Warning: unable to determine stain for file {filename}")
                continue

            # Prepare new filename and destination paths
            # new_filename = f"{sample_name}_{channel}.tif"
            new_filename = filename

            # Copy to stain and image name folders
            source_path = os.path.join(input_folder, filename)
            shutil.copy(source_path, os.path.join(output_folder, channel, new_filename))

    print("File sorting and renaming complete.")

    return output_folder

if __name__ == "__main__":
    input_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\train\image'
    sort_and_clean_filenames(input_folder)