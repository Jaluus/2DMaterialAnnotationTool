from PIL import Image
import imagehash
import os
import cv2

import matplotlib.pyplot as plt

IMAGE_DIR = r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images\Graphene\Set2_no_Duplicates\Images_no_vignette"
MASK_DIR = IMAGE_DIR.replace("Images_no_vignette", "Masks")
IMAGE_NAMES = [name for name in os.listdir(IMAGE_DIR) if name.endswith(".png")]

stored_hashes = {}

for image_name in IMAGE_NAMES:
    mask_name = "Mask_" + image_name
    image_path = os.path.join(IMAGE_DIR, image_name)
    mask_path = os.path.join(MASK_DIR, mask_name)

    image = Image.open(image_path)
    mask = cv2.imread(mask_path, 0)

    hash = imagehash.average_hash(image)

    if hash in stored_hashes.keys():
        print(image_path)

        # Append the mask to the existing mask
        original_mask_path = os.path.join(MASK_DIR, "Mask_" + stored_hashes[hash])
        original_mask = cv2.imread(original_mask_path, 0)
        new_mask = cv2.bitwise_or(original_mask, mask)
        cv2.imwrite(original_mask_path, new_mask)

        os.remove(image_path)
        os.remove(mask_path)
        continue

    stored_hashes[hash] = image_name
