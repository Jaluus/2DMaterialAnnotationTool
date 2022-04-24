from calendar import c
import cv2
import numpy as np
import os

IMAGE_DIR = r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images\Graphene\Set2_no_Duplicates\Images_no_vignette"
MASK_DIR = IMAGE_DIR.replace("Images_no_vignette", "Masks")
IMAGE_NAMES = [name for name in os.listdir(IMAGE_DIR) if name.endswith(".png")]
IMAGE_PATHS = [os.path.join(IMAGE_DIR, name) for name in IMAGE_NAMES]
MASK_NAMES = ["Mask_" + name for name in os.listdir(IMAGE_DIR) if name.endswith(".png")]
MASK_PATHS = [os.path.join(MASK_DIR, name) for name in MASK_NAMES]

current_idx = 0
current_image = cv2.imread(IMAGE_PATHS[current_idx])
current_mask = cv2.imread(MASK_PATHS[current_idx])
current_image_overlay = current_image.copy()
current_image_overlay = cv2.addWeighted(current_image, 1, current_mask, 0.5, 0)


def update_current_image():
    global current_image, current_mask, current_image_overlay
    current_image = cv2.imread(IMAGE_PATHS[current_idx])
    current_mask = cv2.imread(MASK_PATHS[current_idx])
    current_image_overlay = current_image.copy()
    current_image_overlay = cv2.addWeighted(current_image, 0.5, current_mask, 0.5, 0)


cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)

cv2.setWindowTitle("Annotator", IMAGE_PATHS[current_idx])

while True:

    cv2.imshow("Annotator", current_image_overlay)
    cv2.setWindowTitle("Annotator", IMAGE_PATHS[current_idx])

    key = cv2.waitKey(1)

    if key == 27:
        break

    if key == ord("d"):
        if current_idx < len(IMAGE_PATHS) - 1:
            current_idx += 1
            update_current_image()

    if key == ord("a"):
        if current_idx > 0:
            current_idx -= 1
            update_current_image()


cv2.destroyAllWindows()
