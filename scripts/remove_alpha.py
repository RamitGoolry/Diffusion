import chunk
from dataclasses import field
from os import remove
import numpy as np
from PIL import Image
from icecream import ic

from pathlib import Path

def remove_alpha(filepath):
    img = Image.open(filepath)
    img = np.array(img)

    img = img[:, :, :3]
    img = Image.fromarray(img)

    img.save(filepath)

DATA_FOLDER = Path('./data/pokemon')

for child in DATA_FOLDER.iterdir():
    remove_alpha(child)