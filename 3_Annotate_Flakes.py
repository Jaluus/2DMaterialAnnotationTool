import cv2
import numpy as np
import os

ANNOTATED_IMAGES_DIR = r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images"

SET_NAME = "Set_10"
MATERIAL = "Graphene"

SET_DIR = os.path.join(ANNOTATED_IMAGES_DIR, MATERIAL, SET_NAME)

IMAGE_DIR = os.path.join(SET_DIR, "Images")
MASK_SAVE_DIR = os.path.join(SET_DIR, "Masks_Human_Annotated")
IMAGE_NAMES = [name for name in os.listdir(IMAGE_DIR) if name.endswith(".png")]
IMAGE_PATHS = [os.path.join(IMAGE_DIR, name) for name in IMAGE_NAMES]

current_idx = 0
marks_updated = False

l_mouse_down = False
r_mouse_down = False

current_image = cv2.imread(IMAGE_PATHS[current_idx])
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
    global current_image_marked, marker_image, watershed_segments, current_image_display
    current_image_marked = np.copy(current_image)
    current_image_display = np.copy(current_image)
    marker_image = np.zeros(current_image.shape[0:2], dtype=np.int32)
    watershed_segments = np.zeros(current_image.shape, dtype=np.uint8)


def update_current_image():
    global current_image
    current_image = cv2.imread(IMAGE_PATHS[current_idx])
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
        mask = np.zeros(current_image.shape[:2], dtype=np.uint8)
        mask[marker_image_copy == 1] = 255
        cv2.imwrite(f"{MASK_SAVE_DIR}/Mask_{IMAGE_NAMES[current_idx]}", mask)

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
        cv2.watershed(current_image, marker_image_copy)

        watershed_segments = np.zeros(current_image.shape, dtype=np.uint8)

        watershed_segments[marker_image_copy == 1] = [0, 0, 255]
        watershed_segments[marker_image_copy == 2] = 0

        watershed_segments = cv2.morphologyEx(
            watershed_segments, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)
        )

        current_image_display = cv2.addWeighted(
            current_image_display, 1, watershed_segments, 1, 0
        )

        marks_updated = False

cv2.destroyAllWindows()
