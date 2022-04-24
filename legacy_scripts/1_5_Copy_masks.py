import cv2
import numpy as np
import os
import shutil

IMAGE_DIR = r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images\Graphene\Set2_no_Duplicates\Images_no_vignette"
NEW_MASK_DIR = r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images\Graphene\Set2_no_Duplicates\Mask_Human_annot"
IMAGE_NAMES = [name for name in os.listdir(IMAGE_DIR) if name.endswith(".png")]
IMAGE_PATHS = [os.path.join(IMAGE_DIR, name) for name in IMAGE_NAMES]

current_idx = 0
marks_updated = False

for i in range(136):
    mask_name = "Mask_" + IMAGE_NAMES[i]

    shutil.copy(
        os.path.join("Masks", mask_name),
        os.path.join(NEW_MASK_DIR, mask_name),
    )
