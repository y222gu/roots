
import os
import glob
import shutil

# === USER CONFIGURATION ===
# Folder containing your original .txt YOLO-seg files:
INPUT_DIR  = r'C:\Users\Yifei\Documents\data_for_publication\test\Zeiss'

# Where to write your remapped files.
# If you want to overwrite in-place, set IN_PLACE = True instead.
OUTPUT_DIR = r'C:\Users\Yifei\Documents\data_for_publication\test\Zeiss_corrected'

# If True, originals in INPUT_DIR will be overwritten (no OUTPUT_DIR used)
IN_PLACE   = False
# ===========================

# define your old→new class mapping here
CLASS_MAP = {
    2: 0,  # stele boundary → 0
    0: 2,  # outer contour  → 2
    1: 1,  # endodermis     → 1
    3: 3,  # aerenchyma     → 3
}

def remap_file(src_path: str, dst_path: str):
    """Read src_path, remap each line’s class id, write to dst_path."""
    with open(src_path, 'r') as src, open(dst_path, 'w') as dst:
        for line in src:
            parts = line.strip().split()
            if not parts:
                dst.write("\n")
                continue
            old_cls = int(parts[0])
            new_cls = CLASS_MAP.get(old_cls, old_cls)
            dst.write(" ".join([str(new_cls)] + parts[1:]) + "\n")

def main():
    # collect all .txt files
    txt_files = glob.glob(os.path.join(INPUT_DIR, '*.txt'))
    if not txt_files:
        print(f"No .txt files found in {INPUT_DIR!r}")
        return

    # determine output folder
    if IN_PLACE:
        print("** Overwriting in-place in:", INPUT_DIR)
        os.makedirs(INPUT_DIR, exist_ok=True)
        out_dir = INPUT_DIR
    else:
        out_dir = OUTPUT_DIR
        os.makedirs(out_dir, exist_ok=True)
        print("Writing remapped files to:", out_dir)

    # process each file
    for src in txt_files:
        fname = os.path.basename(src)
        dst = os.path.join(out_dir, fname)
        # optionally back up originals if overwriting
        # if IN_PLACE:
        #     shutil.copy2(src, src + '.bak')
        remap_file(src, dst)
        action = "overwritten" if IN_PLACE else f"-> {dst}"
        print(f"  • {fname} {action}")

if __name__ == '__main__':
    main()