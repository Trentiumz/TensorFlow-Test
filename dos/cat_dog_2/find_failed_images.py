import PIL
from PIL import Image
from os import listdir
import numpy as np
dirs = ["Test", "Train", "Val"]
for cur_dir in dirs:
    for extra in ["Cat", "Dog"]:
        cur_path = f"./Data/{cur_dir}/{extra}/"
        for file_name in listdir(cur_path):
            try:
                img = Image.open(cur_path + file_name)
                np_img = np.array(img)
            except PIL.UnidentifiedImageError:
                print(cur_dir, extra, file_name)