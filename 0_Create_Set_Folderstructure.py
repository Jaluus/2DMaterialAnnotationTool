import os

SET_DIR = r"C:\Users\Uslu.INSTITUT2B\Desktop\Annotated_Images\Graphene"

current_sets = os.listdir(SET_DIR)
max_set_idx = max([int(set_name.split("_")[-1]) for set_name in current_sets])

os.mkdir(os.path.join(SET_DIR, f"Set_{max_set_idx + 1}"))

new_set_path = os.path.join(SET_DIR, f"Set_{max_set_idx + 1}")

os.mkdir(os.path.join(new_set_path, "Images"))
os.mkdir(os.path.join(new_set_path, "Detected_Images"))
os.mkdir(os.path.join(new_set_path, "Masks_Human_Annotated"))
os.mkdir(os.path.join(new_set_path, "Plots"))
