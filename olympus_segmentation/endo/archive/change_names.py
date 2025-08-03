#!/usr/bin/env python3
import os
import sys

def rename_annotation_files(root_dir):
    # List all annotation files up-front
    txt_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.txt')]

    for entry in os.listdir(root_dir):
        subdir = os.path.join(root_dir, entry)
        # only consider directories
        if not os.path.isdir(subdir):
            continue

        # look for a .txt starting with the folder name + '_'
        prefix = entry + '_'
        matches = [f for f in txt_files if f.startswith(prefix)]
        if not matches:
            print(f"[!] no .txt matched for folder '{entry}'")
            continue

        # if there are multiple matches, skip (or you could pick the first)
        if len(matches) > 1:
            print(f"[!] multiple matches for '{entry}': {matches}")
            continue

        old_name = matches[0]
        new_name = entry + '.txt'
        old_path = os.path.join(root_dir, old_name)
        new_path = os.path.join(root_dir, new_name)

        if os.path.exists(new_path):
            print(f"[!] target '{new_name}' already exists; skipping")
        else:
            os.rename(old_path, new_path)
            print(f"Renamed '{old_name}' → '{new_name}'")

        txt_files.remove(old_name)


if __name__ == '__main__':
    # default to your folder, or take an override from the command line
    DEFAULT_ROOT = r"C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\Images_SRN"
    if len(sys.argv) == 2:
        root = sys.argv[1]
    else:
        root = DEFAULT_ROOT
        print(f"No folder argument given; defaulting to\n    {root!r}")

    if not os.path.isdir(root):
        print(f"Error: '{root}' is not a directory")
        sys.exit(1)

    rename_annotation_files(root)
