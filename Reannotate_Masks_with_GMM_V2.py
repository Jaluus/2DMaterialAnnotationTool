import cv2
import numpy as np
import os
import json

from sympy import centroid

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

        # only remove the small components if there are any
        mask = np.where(np.isin(labels, small_components), 0, mask)

    return mask


classifier = detector_class(gmm_params, standard_deviation_threshold=5)

current_idx = 0
marks_updated = False

l_mouse_down = False
r_mouse_down = False

current_image = remove_vignette(cv2.imread(IMAGE_PATHS[current_idx]))
current_image_contrast = classifier.create_contrast_image(current_image)
full_flake_mask = process_mask(classifier.create_flake_mask(current_image_contrast))
current_image_marked = np.copy(current_image)
current_image_display = np.copy(current_image)
marker_image = np.zeros(current_image.shape[:2], dtype=np.int32)
watershed_segments = np.zeros(current_image.shape, dtype=np.uint8)


def mouse_callback(event, x, y, flags, param):
    global marks_updated, l_mouse_down, r_mouse_down
    if event == cv2.EVENT_LBUTTONDOWN:
        l_mouse_down = True

    if event == cv2.EVENT_LBUTTONUP:
        l_mouse_down = False

    if event == cv2.EVENT_RBUTTONDOWN:
        r_mouse_down = True

    if event == cv2.EVENT_RBUTTONUP:
        r_mouse_down = False

    if l_mouse_down:
        cv2.circle(marker_image, (x, y), 3, 1, -1)
        cv2.circle(current_image_marked, (x, y), 3, (0, 255, 0), -1)
        marks_updated = True

    if r_mouse_down:
        cv2.circle(marker_image, (x, y), 3, 2, -1)
        cv2.circle(current_image_marked, (x, y), 3, (0, 0, 255), -1)
        marks_updated = True


def clear_marks():
    global current_image_marked, marker_image, watershed_segments, current_image_display, watershed_was_running
    current_image_marked = np.copy(current_image)
    current_image_display = np.copy(current_image)
    marker_image = np.zeros(current_image.shape[0:2], dtype=np.int32)
    watershed_segments = np.zeros(current_image.shape, dtype=np.uint8)
    watershed_was_running = False


def update_current_image():
    global current_image, current_image_contrast, full_flake_mask
    current_image = cv2.imread(IMAGE_PATHS[current_idx])
    current_image = remove_vignette(current_image, flatfield)
    current_image_contrast = classifier.create_contrast_image(current_image)
    full_flake_mask = classifier.create_flake_mask(current_image_contrast)
    full_flake_mask = process_mask(full_flake_mask)
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

        mask = np.where(
            marker_image_copy > len(gmm_params.keys()), 0, marker_image_copy
        )

        cv2.imwrite(
            f"{MASK_SAVE_DIR}/Mask_{IMAGE_NAMES[current_idx]}",
            mask.astype(np.uint8),
        )

    if key == ord("d"):
        if current_idx < len(IMAGE_PATHS) - 1:
            current_idx += 1
            update_current_image()

    if key == ord("a"):
        if current_idx > 0:
            current_idx -= 1
            update_current_image()

    # If we clicked somewhere, call the watershed algorithm on our chosen markers
    if marks_updated:

        current_image_display = current_image_marked.copy()
        marker_image_copy = marker_image.copy()

        # run the watershed algorithm with the chosen markers
        cv2.watershed(current_image, marker_image_copy)

        # create a mask of the watershed segments
        watershed_segments = np.zeros(current_image.shape[:2], dtype=np.uint8)
        watershed_segments[marker_image_copy == 2] = 0
        watershed_segments[marker_image_copy == 1] = 255

        # to illustrate the watershed algorithm, we'll overlay the watershed segments with a gradient
        watershed_segments_expanded = cv2.dilate(watershed_segments, np.ones((5, 5)))
        watershed_segments_grad = cv2.morphologyEx(
            watershed_segments_expanded,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )

        full_flake_mask_cropped = np.where(watershed_segments > 0, full_flake_mask, 0)
        # illustrate the gradient
        current_image_display[watershed_segments_grad > 0] = [255, 255, 255]

        for i in range(1, MAX_LAYER_THICKNESS + 1):
            current_image_display[full_flake_mask_cropped == i] = (
                0.25 * COLORS[i]
                + 0.75 * current_image_display[full_flake_mask_cropped == i]
            )

        marks_updated = False

cv2.destroyAllWindows()
