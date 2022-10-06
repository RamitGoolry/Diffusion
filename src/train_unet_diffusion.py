from modules.unet import UNet
from diffusion_utils import add_gaussian_noise
from modules.vqgan_utils import load_data

import torch

# =========================================
#     Data Structures Used
# =========================================

class Stack:
    def __init__(self):
        self.stack = []
    
    def push(self, item):
        self.stack.append(item)
    
    def pop(self):
        item = self.stack[-1]
        self.stack = self.stack[:-1]
        return item

class dotdict(dict):
    """
    Dot notation to access dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



# TODO List
# - Write a Trainer Class that uses wandb
# - For every image, it adds diffusion noise to the image and adds that to the stack
# - Then we pop from the stack, and learn to "Undiffuse the noise" from the UNet

config = dotdict({
    'features' : (16, 32, 64),
    'lr' : 2.5e-5,
    'epochs' : 50,
    'image_size' : 128
})

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class UNetDiffusionTrainer:
    def __init__(self, config, device):
        self.model = UNet(features = config.features)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = config.lr)

    def train()