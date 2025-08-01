
import os
import re
import shutil

# === USER CONFIGURATION ===
INPUT_DIR =r'C:\Users\Yifei\Documents\data_for_publication\train\C10\Sorghum\Images_SRN' # path to the folder containing your date-folders
# ===========================

def extract_code(folder_name):
    # try regex _<LETTER><DIGITS>
    m = re.search(r'_(?P<code>[A-Z]\d{1,2})', folder_name)
    if m:
        return m.group('code')
    # fallback: take the 3rd underscore-field, then first token
    parts = folder_name.split('_', 2)
    if len(parts) >= 3:
        return parts[2].split()[0]
    # if all else fails, return original name
    return folder_name

def extract_channel(file_name):
    base = os.path.splitext(file_name)[0]
    # look for a known channel keyword
    for ch in ('DAPI','GFP','TRITC'):
        if re.search(rf'\b{ch}\b', base, re.IGNORECASE):
            return ch
    # fallback: last underscore token
    return base.split('_')[-1]

def main():
    for old_folder in os.listdir(INPUT_DIR):
        old_path = os.path.join(INPUT_DIR, old_folder)
        if not os.path.isdir(old_path):
            continue

        code = extract_code(old_folder)
        new_path = os.path.join(INPUT_DIR, code)

        # rename folder (if target exists, we'll still process inside it)
        if old_path != new_path:
            if os.path.exists(new_path):
                print(f"[!] Target folder already exists, skipping rename: {new_path}")
            else:
                os.rename(old_path, new_path)
                print(f"Renamed folder: {old_folder} → {code}")

        # process files inside
        for fname in os.listdir(new_path):
            src = os.path.join(new_path, fname)
            if not os.path.isfile(src):
                continue
            root, ext = os.path.splitext(fname)
            if ext.lower() not in ('.tif', '.tiff'):
                continue

            channel = extract_channel(fname)
            new_fname = f"{code}_{channel}{ext}"
            dst = os.path.join(new_path, new_fname)

            if src == dst:
                continue
            if os.path.exists(dst):
                print(f"  [!] File exists, skipping: {new_fname}")
                continue

            os.rename(src, dst)
            print(f"  Renamed file: {fname} → {new_fname}")

    print("All done.")

if __name__ == '__main__':
    main()
