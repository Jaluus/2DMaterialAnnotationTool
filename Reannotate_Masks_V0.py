from calendar import c
import sys
import cv2
import numpy as np
import os
import sys

ANNOTATED_IMAGES_DIR = r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images"

SET_NAME = "Set_10"
MATERIAL = "Graphene"

SET_DIR = os.path.join(ANNOTATED_IMAGES_DIR, MATERIAL, SET_NAME)

IMAGE_DIR = os.path.join(SET_DIR, "Images")
SPLICE_MASK_DIR = os.path.join(SET_DIR, "Masks_Full")
MASK_NAMES = [name for name in os.listdir(SPLICE_MASK_DIR) if name.endswith(".png")]
MASK_DIRS = [folder for folder in os.listdir(SET_DIR) if folder[-1].isnumeric()]
NUM_THICKNESS = len(MASK_DIRS)
COLORS = [
    [255, 255, 255],
    [0, 0, 255],
    [0, 255, 0],
    [0, 255, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
    [255, 255, 0],
    [255, 0, 255],
]

# create the new mask folder if it doesn't exist
if not os.path.exists(SPLICE_MASK_DIR):
    sys.exit(f"{SPLICE_MASK_DIR} does not exist")

current_idx = 0
marks_updated = False

# 255 means the background, 1 - n means the thickness
current_marker = 1

l_mouse_down = False
r_mouse_down = False
l_mouse_click = False
r_mouse_click = False
mouse_x = 0
mouse_y = 0

current_image = cv2.imread(os.path.join(IMAGE_DIR, MASK_NAMES[current_idx]))
current_mask = cv2.imread(os.path.join(SPLICE_MASK_DIR, MASK_NAMES[current_idx]), 0)
marker_mask = np.zeros(current_mask.shape, dtype=np.int32)


def mark_mask_on_image(image, mask):
    colored_mask = np.zeros(image.shape, dtype=np.uint8)

    for i in range(1, NUM_THICKNESS):
        grad_mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        grad_mask = cv2.dilate(
            grad_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )

        colored_mask[grad_mask == i] = COLORS[i]

    marked_image = cv2.addWeighted(image, 1, colored_mask, 1, 0)

    return marked_image


def mark_watershed_marker_on_image(image, mask):
    colored_mask = np.zeros(image.shape, dtype=np.uint8)

    for i in range(1, NUM_THICKNESS):
        colored_mask[mask == i] = COLORS[i]

    colored_mask[mask == 255] = COLORS[0]

    marked_image = cv2.addWeighted(image, 1, colored_mask, 1, 0)

    return marked_image


def mouse_callback(event, x, y, flags, param):
    global marks_updated, l_mouse_down, r_mouse_down, mouse_x, mouse_y, r_mouse_click, l_mouse_click

    mouse_x = x
    mouse_y = y

    r_mouse_click = False
    l_mouse_click = False

    if event == cv2.EVENT_LBUTTONDOWN:
        l_mouse_down = True
        l_mouse_click = True

    if event == cv2.EVENT_LBUTTONUP:
        l_mouse_down = False

    if event == cv2.EVENT_RBUTTONDOWN:
        r_mouse_down = True
        r_mouse_click = True

    if event == cv2.EVENT_RBUTTONUP:
        r_mouse_down = False

    if l_mouse_down:
        marks_updated = True

    if r_mouse_down:
        marks_updated = True


def reload_current_image():
    global current_image, current_mask, displayed_image
    current_image = cv2.imread(os.path.join(IMAGE_DIR, MASK_NAMES[current_idx]))
    current_mask = cv2.imread(os.path.join(SPLICE_MASK_DIR, MASK_NAMES[current_idx]), 0)
    displayed_image = mark_mask_on_image(current_image, current_mask)


def remove_region(x, y):
    global current_mask, displayed_image

    # get the current clicked thickness and generate a sub-mask with only that thickness
    clicked_thickness = current_mask[y, x]
    thickness_mask = np.where(current_mask == clicked_thickness, 1, 0).astype(np.uint8)

    # extract all the connected components from the thickness mask
    _, connected_components = cv2.connectedComponents(thickness_mask, connectivity=4)

    # get the connected component which was clicked
    conn_mask = np.where(connected_components == connected_components[y, x], 1, 0)

    # remove the connected component
    current_mask = np.where(conn_mask == 1, 0, current_mask)

    update_current_displayed_image()


def set_watershed_marker(x, y):
    global marker_mask, current_marker
    cv2.circle(marker_mask, (x, y), 3, current_marker, -1)
    update_current_displayed_image()


def update_current_displayed_image():
    global displayed_image, current_image, current_mask, marker_mask
    # update the displayed image
    displayed_image = mark_mask_on_image(current_image, current_mask)
    displayed_image = mark_watershed_marker_on_image(displayed_image, marker_mask)


cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Annotator", mouse_callback)
cv2.setWindowTitle("Annotator", MASK_NAMES[current_idx])

displayed_image = mark_mask_on_image(current_image, current_mask)

while True:

    displayed_image_mouse = cv2.circle(
        displayed_image.copy(), (mouse_x, mouse_y), 3, (0, 0, 255), -1
    )

    cv2.imshow("Annotator", displayed_image_mouse)
    cv2.setWindowTitle(
        "Annotator",
        f"{MASK_NAMES[current_idx]} | {current_idx + 1 }/{len(MASK_NAMES)}",
    )

    key = cv2.waitKey(10)

    # if key == ord("1"):
    #     current_marker = 1

    # if key == ord("2"):
    #     current_marker = 255

    # remove the region in the mask over which the mouse is hovering
    if r_mouse_click:
        remove_region(mouse_x, mouse_y)

    # if l_mouse_click:
    #     set_watershed_marker(mouse_x, mouse_y)

    if key == 27:
        break

    if key == ord("s"):
        cv2.imwrite(
            os.path.join(SPLICE_MASK_DIR, MASK_NAMES[current_idx]), current_mask
        )

    # save a blank image as the mask
    if key == ord("x"):
        current_mask = np.zeros(current_image.shape[:2], dtype=np.uint8)
        cv2.imwrite(
            os.path.join(SPLICE_MASK_DIR, MASK_NAMES[current_idx]), current_mask
        )

    if key == ord("d"):
        if current_idx < len(MASK_NAMES) - 1:
            current_idx += 1
            reload_current_image()

    if key == ord("a"):
        if current_idx > 0:
            current_idx -= 1
            reload_current_image()

    if key == ord("x"):
        # delete the current mask
        pass

    # # # If we clicked somewhere, call the watershed algorithm on our chosen markers
    # if key == ord("r"):

    #     # copy all the markers to the mask
    #     watershed_segments = marker_mask.copy()

    #     # copy the current mask to the watershed mask
    #     for i in range(1, np.max(current_mask) + 1):
    #         watershed_segments[current_mask == i] = i

    #     # run the watershed algorithm with the chosen markers
    #     cv2.watershed(current_image, watershed_segments)

    #     # convert the watershed result to a mas

    #     current_mask = np.where(
    #         watershed_segments.astype(np.uint8) == 255,
    #         0,
    #         watershed_segments.astype(np.uint8),
    #     )

    #     update_current_displayed_image()

cv2.destroyAllWindows()
