import os
import shutil
import uuid

SCAN_PATH = r"C:\Users\Uslu.INSTITUT2B\Desktop\Mikroskop_Bilder\Graphen_scan_katarina_21032022"

SAVE_DIR = "Images"

CHIP_DIRS = [
    name
    for name in os.listdir(SCAN_PATH)
    if os.path.isdir(os.path.join(SCAN_PATH, name))
    if name.startswith("Chip")
]

ctr = 0

for chip_dir in CHIP_DIRS:
    flake_dirs = [
        name
        for name in os.listdir(os.path.join(SCAN_PATH, chip_dir))
        if os.path.isdir(os.path.join(SCAN_PATH, chip_dir, name))
        if name.startswith("Flake")
    ]

    for flake_dir in flake_dirs:
        ctr += 1
        image_path = os.path.join(SCAN_PATH, chip_dir, flake_dir, "raw_img.png")
        mask_path = os.path.join(SCAN_PATH, chip_dir, flake_dir, "flake_mask.png")
        shutil.copy(image_path, SAVE_DIR)
        shutil.copy(mask_path, "Masks")

        unique_filename = str(uuid.uuid4())
        
        os.rename(
            os.path.join(SAVE_DIR, "raw_img.png"),
            os.path.join(SAVE_DIR, f"{unique_filename}.png"),
        )
        
        os.rename(
            os.path.join("Masks", "flake_mask.png"),
            os.path.join("Masks", f"Mask_{unique_filename}.png"),
        )

        