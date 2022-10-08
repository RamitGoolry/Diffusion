import os
import albumentations

import numpy as np
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImagePaths(Dataset):
    def __init__(self, path, size = None):
        self.size = size
        
        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(width=self.size, height=self.size)
        self.preprocessor = albumentations.Compose([
            self.rescaler, self.cropper
        ])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image = image)['image']
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)

        return image

    def __getitem__(self, i):
        return self.preprocess_image(self.images[i]) 

def load_data(config):
    train_data = ImagePaths(config.dataset_path, size=256)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    return train_loader

# -------------------------------------------- #
#               Module Utils
#          for Encoder, Decoder etc.
# -------------------------------------------- #

def weights_init(m : nn.Module):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)