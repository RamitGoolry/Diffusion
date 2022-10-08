#! /usr/bin/env python3

import os
from PIL import Image
from pathlib import Path

from icecream import ic

SOURCE_DIR = './data/pokemon'
DEST_DIR = './data_128x128/pokemon'

def convert(source_path : str) -> bool:
    ic(source_path)

    img = Image.open(source_path)
    img = img.resize(size = (img.width // 2,  img.height // 2), resample=Image.Resampling.BICUBIC)
    img.save(source_path.replace(SOURCE_DIR, DEST_DIR))
    return True

def main():
    for path in os.listdir(SOURCE_DIR):
        convert(os.path.join(SOURCE_DIR, path))

if __name__ == '__main__':
    main()
