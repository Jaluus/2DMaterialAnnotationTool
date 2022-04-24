from PIL import Image
import imagehash
import os

IMAGE_DIR = "Images"
IMAGE_NAMES = [name for name in os.listdir(IMAGE_DIR) if name.endswith(".png")]
IMAGE_PATHS = [os.path.join(IMAGE_DIR, name) for name in IMAGE_NAMES]

stored_hashes = []

for image_path in IMAGE_PATHS:
    image = Image.open(image_path)
    hash = imagehash.average_hash(image)

    if hash in stored_hashes:
        print(image_path)
        os.remove(image_path)
        continue

    stored_hashes.append(hash)
