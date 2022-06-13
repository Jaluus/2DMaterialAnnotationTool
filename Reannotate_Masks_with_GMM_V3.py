import cv2
import numpy as np
import os
import json

from scripts.detection_class import detector_class
from scripts.preprocessor_functions import remove_vignette

ANNOTATED_IMAGES_DIR = r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images"

SET_NAME = "Set_10"
MATERIAL = "Graphene"

SET_DIR = os.path.join(ANNOTATED_IMAGES_DIR, MATERIAL, SET_NAME)

SET_DIR = os.path.join(ANNOTATED_IMAGES_DIR, MATERIAL, SET_NAME)

IMAGE_DIR = os.path.join(SET_DIR, "Images")
FLAT_FIELD_PATH = os.path.join(SET_DIR, "flatfield.png")
MASK_SAVE_DIR = os.path.join(SET_DIR, "Masks_Full_Human_Annotated")
IMAGE_NAMES = [name for name in os.listdir(IMAGE_DIR) if name.endswith(".png")]
IMAGE_PATHS = [os.path.join(IMAGE_DIR, name) for name in IMAGE_NAMES]
GMM_PARAMS = os.path.join(SET_DIR, "gaussian_mixture_model_contrast_data.json")
COLORS = np.array(
    [
        [255, 255, 255],
        [0, 0, 255],
        [0, 255, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [255, 0, 0],
    ]
)

MAX_LAYER_THICKNESS = 4

if not os.path.exists(MASK_SAVE_DIR):
    os.makedirs(MASK_SAVE_DIR)

flatfield = cv2.imread(FLAT_FIELD_PATH)

with open(GMM_PARAMS, "r") as f:
    gmm_params = json.load(f)


def process_mask(mask, min_size=40):

    mask[mask > MAX_LAYER_THICKNESS] = 0

    for i in range(1, MAX_LAYER_THICKNESS + 1):
        submask = np.where(mask == i, 1, 0).astype(np.uint8)

        # extract all the connected components from the thickness mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            submask, connectivity=4
        )

        # TODO: Speedup the following loop
        # remove all the connected components that are smaller than the minimum size
        small_components = np.argwhere(stats[:, 4] < min_size)

        print(np.isin(labels, small_components))

        # only remove the small components if there are any
        mask = np.where(np.isin(labels, small_components), 0, mask)

    return mask


classifier = detector_class(gmm_params, standard_deviation_threshold=5)

current_idx = 103
thickness_shown = True

l_mouse_down = False
r_mouse_down = False

current_image = remove_vignette(cv2.imread(IMAGE_PATHS[current_idx]), flatfield)
current_image_contrast = classifier.create_contrast_image(current_image)
full_flake_mask = process_mask(classifier.create_flake_mask(current_image_contrast))
original_full_flake_mask = np.copy(full_flake_mask)
erased_mask = np.zeros(current_image.shape[:2], dtype=np.uint8)
current_image_display = np.copy(current_image)


def mouse_callback(event, x, y, flags, param):
    global erased_mask, l_mouse_down, r_mouse_down
    if event == cv2.EVENT_LBUTTONDOWN:
        l_mouse_down = True

    if event == cv2.EVENT_LBUTTONUP:
        l_mouse_down = False

    if event == cv2.EVENT_RBUTTONDOWN:
        r_mouse_down = True

    if event == cv2.EVENT_RBUTTONUP:
        r_mouse_down = False

    if l_mouse_down:
        cv2.circle(erased_mask, (x, y), 5, 1, -1)

    if r_mouse_down:
        remove_region(x, y)


def remove_region(x, y):
    global full_flake_mask

    # get the current clicked thickness and generate a sub-mask with only that thickness
    clicked_thickness = full_flake_mask[y, x]
    thickness_mask = np.where(full_flake_mask == clicked_thickness, 1, 0).astype(
        np.uint8
    )

    # extract all the connected components from the thickness mask
    _, connected_components = cv2.connectedComponents(thickness_mask, connectivity=4)

    # get the connected component which was clicked
    conn_mask = np.where(connected_components == connected_components[y, x], 1, 0)

    # remove the connected component
    full_flake_mask = np.where(conn_mask == 1, 0, full_flake_mask)


def clear_marks():
    global current_image_display, erased_mask, full_flake_mask
    current_image_display = np.copy(current_image)
    erased_mask = np.zeros(current_image.shape[:2], dtype=np.uint8)
    full_flake_mask = np.copy(original_full_flake_mask)


def update_current_image():
    global current_image, current_image_contrast, full_flake_mask, original_full_flake_mask
    current_image = remove_vignette(cv2.imread(IMAGE_PATHS[current_idx]), flatfield)
    current_image_contrast = classifier.create_contrast_image(current_image)
    full_flake_mask = process_mask(classifier.create_flake_mask(current_image_contrast))
    original_full_flake_mask = np.copy(full_flake_mask)
    clear_marks()


cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Annotator", mouse_callback)

cv2.setWindowTitle("Annotator", IMAGE_PATHS[current_idx])

while True:

    cv2.imshow("Annotator", current_image_display)
    cv2.setWindowTitle(
        "Annotator",
        f"{IMAGE_NAMES[current_idx]} | {current_idx + 1 }/{len(IMAGE_PATHS)}",
    )

    key = cv2.waitKey(10)

    if key == 27:
        break

    elif key == ord("c"):
        clear_marks()

    if key == ord("s"):
        cv2.imwrite(
            os.path.join(MASK_SAVE_DIR, IMAGE_NAMES[current_idx]), full_flake_mask
        )

    if key == ord("d"):
        if current_idx < len(IMAGE_PATHS) - 1:
            current_idx += 1
            update_current_image()

    if key == ord("a"):
        if current_idx > 0:
            current_idx -= 1
            update_current_image()

    if key == ord("w"):
        thickness_shown = not thickness_shown

    if thickness_shown:
        current_image_display = np.copy(current_image)
        for i in range(1, MAX_LAYER_THICKNESS + 1):
            current_image_display[full_flake_mask == i] = (
                1 * COLORS[i] + 0.0 * current_image_display[full_flake_mask == i]
            )
        # current_image_display[erased_mask == 1] = [0, 0, 0]

    else:
        current_image_display = current_image

cv2.destroyAllWindows()
