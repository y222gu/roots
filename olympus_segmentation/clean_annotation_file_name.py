import os
import glob

data_folder = r'C:\Users\Yifei\Documents\new_endo_model_training\val\annotation'

list_of_file_paths = glob.glob(os.path.join(data_folder, '**/*.txt'), recursive=True)

for file_path in list_of_file_paths:
    new_file_name = file_path.split('_raw')[0]
    new_file_name = new_file_name + '.txt'
    os.rename(file_path, new_file_name)
