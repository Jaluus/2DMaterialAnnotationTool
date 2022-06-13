import cv2
import numpy as np
import os
import shutil

ANNOTATED_IMAGES_DIR = r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images"

SET_NAME = f"Set_10"
MATERIAL = "Graphene"

SET_DIR = os.path.join(ANNOTATED_IMAGES_DIR, MATERIAL, SET_NAME)

IMAGE_DIR = os.path.join(SET_DIR, "Images")
SPLICE_MASK_DIR = os.path.join(SET_DIR, "Masks_Full")
IMAGE_NAMES = [name for name in os.listdir(IMAGE_DIR) if name.endswith(".png")]

# create the new mask folder if it doesn't exist
if not os.path.exists(SPLICE_MASK_DIR):
    os.makedirs(SPLICE_MASK_DIR)

# get all the mask folders
# each folder corresponds to a different thickness
MASK_DIRS = [folder for folder in os.listdir(SET_DIR) if folder[-1].isnumeric()]

for idx, image_name in enumerate(IMAGE_NAMES):
    print(f"{idx} / {len(IMAGE_NAMES)}")
    image = cv2.imread(os.path.join(IMAGE_DIR, image_name))
    mask_full = np.zeros(image.shape[:2], dtype=np.uint8)

    for mask_dir in MASK_DIRS:
        mask_thickness = int(mask_dir.split("_")[-1])
        mask_dir_path = os.path.join(SET_DIR, mask_dir)
        mask_names = [
            os.path.join(mask_dir_path, name)
            for name in os.listdir(mask_dir_path)
            if name.endswith(image_name)
        ]

        for mask_name in mask_names:
            mask_path = os.path.join(mask_dir_path, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # set all the pixels in the mask to thickness value
            mask[mask > 0] = mask_thickness

            # add the mask to the full mask
            mask_full = cv2.bitwise_or(mask_full, mask)

    cv2.imwrite(os.path.join(SPLICE_MASK_DIR, image_name), mask_full)
