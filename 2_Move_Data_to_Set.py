import os
import shutil
import uuid

SET_NAME = "Set_11"
MATERIAL = "Graphene"
DATASET_PATH = r"E:\Datasets\Graphene-Taoufiq_Exfol-16-05-2022"
ANNOTATED_IMAGES_PATH = r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images"

DATASET_IMAGE_PATH = os.path.join(DATASET_PATH, "20x")
SET_PATH = os.path.join(ANNOTATED_IMAGES_PATH, MATERIAL, SET_NAME)
MASKED_IMAGES_PATH = os.path.join(DATASET_IMAGE_PATH, "Masked_Images")
ORIGINAL_IMAGES_PATH = os.path.join(DATASET_IMAGE_PATH, "Pictures")

IMAGE_NAMES = [name for name in os.listdir(MASKED_IMAGES_PATH) if name.endswith(".png")]

# copy the meta information of the images
shutil.copy(
    os.path.join(DATASET_PATH, "meta.json"),
    SET_PATH,
)
shutil.copy(
    os.path.join(DATASET_IMAGE_PATH, "Meta", "0.json"),
    SET_PATH,
)

for image_name in IMAGE_NAMES:

    shutil.copy(
        os.path.join(ORIGINAL_IMAGES_PATH, image_name),
        os.path.join(SET_PATH, "Images"),
    )

    unique_filename = str(uuid.uuid4())
    os.rename(
        os.path.join(SET_PATH, "Images", image_name),
        os.path.join(SET_PATH, "Images", f"{unique_filename}.png"),
    )
