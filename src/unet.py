from turtle import forward
import torch
import torch.nn as nn

import torchvision.transforms as T

from icecream import ic

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()
    
    def forward(self, inputs):
        x = self.conv1(inputs)
        # x = self.bn1(x) # TODO
        x = self.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x) # TODO
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels) 
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)

        ic(x.shape, p.shape)

        return x, p

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upscale = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_channels + out_channels, out_channels)

    
    def forward(self, inputs, skip):
        x = self.upscale(inputs)
        ic(inputs.shape, x.shape, skip.shape)

        x = torch.cat([x, skip], axis = 1)
        x = self.conv(x)

        ic(x.shape)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 256 x 256 x 3
        self.block1 = Encoder(3, 16)
        self.block2 = Encoder(16, 32)
        self.block3 = Encoder(32, 64)

        self.block3_inv = Decoder(64, 32)
        self.block2_inv = Decoder(32, 16)
        self.block1_inv = Decoder(16, 3)

        self.relu = nn.ReLU()

    def forward(self, img):
        x1, p1 = self.block1(img)
        x1, p1 = self.relu(x1), self.relu(p1)

        x2, p2 = self.block2(p1)
        x2, p2 = self.relu(x2), self.relu(p2)

        x3, p3 = self.block3(p2)
        x3, p3 = self.relu(x3), self.relu(p3)

        x3_inv = self.block3_inv(x3, p3)

        return x3

if __name__ == '__main__':
    unet = UNet()

    from PIL import Image

    FILEPATH = "./data/pokemon/2.png"
    OUTFILE = "./test.png"

    img = Image.open(FILEPATH)
    img = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])(img)

    ic(img.shape)

    ret = unet(img)