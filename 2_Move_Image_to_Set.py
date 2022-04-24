import os
import shutil
import uuid

DATASET_PATH = r"E:\Datasets\Graphene-Hristiyana_Cleanroom-22-04-2022\20x"
COPY_TO_PATH = (
    r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images\Graphene\Set_7\Images"
)

MASKED_IMAGES_PATH = os.path.join(DATASET_PATH, "Masked_Images")
ORIGINAL_IMAGES_PATH = os.path.join(DATASET_PATH, "Pictures")

IMAGE_NAMES = [name for name in os.listdir(MASKED_IMAGES_PATH) if name.endswith(".png")]


for image_name in IMAGE_NAMES:

    original_image_path = os.path.join(ORIGINAL_IMAGES_PATH, image_name)
    shutil.copy(original_image_path, COPY_TO_PATH)

    unique_filename = str(uuid.uuid4())
    os.rename(
        os.path.join(COPY_TO_PATH, image_name),
        os.path.join(COPY_TO_PATH, f"{unique_filename}.png"),
    )
