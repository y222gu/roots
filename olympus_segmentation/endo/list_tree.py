import os

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

if __name__ == "__main__":
    # Set the path to the directory you want to list
    directory_path = r'C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\Images_SRN'
    list_files(directory_path)