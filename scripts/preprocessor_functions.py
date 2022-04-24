import cv2
import numpy as np


def remove_vignette_legecy(image, flat_field):
    """Removes the Vignette from the Image

    Args:
        image (NxMx3 Array): The Image with the Vignette
        flat_field (NxMx3 Array): the Flat Field in RGB

    Returns:
        (NxMx3 Array): The Image without the Vignette
    """
    # convert to hsv and cast to 16bit, to be able to add more than 255
    image_hsv = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), dtype=np.uint16)
    flat_field_hsv = np.asarray(
        cv2.cvtColor(flat_field, cv2.COLOR_BGR2HSV), dtype=np.uint16
    )

    # get the filter and apply it to the image
    image_hsv[:, :, 2] = (
        image_hsv[:, :, 2]
        / flat_field_hsv[:, :, 2]
        * cv2.mean(flat_field_hsv[:, :, 2])[0]
    )

    # clip it back to 255
    image_hsv[:, :, 2][image_hsv[:, :, 2] > 255] = 255

    # Recast to uint8 as the color depth is 8bit per channel
    image_hsv = np.asarray(image_hsv, dtype=np.uint8)

    # reconvert to bgr
    image_no_vigentte = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    return image_no_vigentte


def remove_vignette(
    image,
    flat_field,
    max_background_value: int = 241,
):
    """Removes the Vignette from the Image

    Args:
        image (NxMx3 Array): The Image with the Vignette
        flat_field (NxMx3 Array): the Flat Field in RGB
        max_background_value (int): the maximum value of the background

    Returns:
        (NxMx3 Array): The Image without the Vignette
    """

    image_no_vigentte = image / flat_field * cv2.mean(flat_field)[:-1]

    image_no_vigentte[image_no_vigentte > max_background_value] = max_background_value

    return np.asarray(image_no_vigentte, dtype=np.uint8)


def calculate_background_color(img, radius=2):

    masks = []

    for i in range(3):
        img_channel = img[:, :, i]

        # A threshold which removes the Unwanted background of the non chip
        # Currently Unused

        hist_r = cv2.calcHist([img_channel], [0], None, [256], [0, 256])

        hist_max_r = np.argmax(hist_r)

        threshed_r = cv2.inRange(
            img_channel, int(hist_max_r - radius), int(hist_max_r + radius)
        )
        background_mask_channel = cv2.erode(threshed_r, np.ones((3, 3)), iterations=3)
        masks.append(background_mask_channel)

    final_mask = cv2.bitwise_and(masks[0], masks[1])
    final_mask = cv2.bitwise_and(final_mask, masks[2])

    return cv2.mean(img, mask=final_mask)[:3]
