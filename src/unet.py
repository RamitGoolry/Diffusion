import torch
import torch.nn as nn

import torchvision.transforms as T
from icecream import ic

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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
        super(Encoder, self).__init__()

        self.conv = ConvBlock(in_channels, out_channels) 
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)

        return x, p

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()

        self.upscale = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_channels + out_channels, out_channels)


    def forward(self, inputs, skip):
        x = self.upscale(inputs)
        x = torch.cat([x, skip], axis = 1)
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self, features = (16, 32, 64), in_channels = 3, out_channels = 3):
        super(UNet, self).__init__()

        # 256 x 256 x 3
        self.encoders = nn.ModuleList() 
        self.decoders = nn.ModuleList()

        for feature in features:
            self.encoders.append(Encoder(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.decoders.append(Decoder(feature * 2, feature))

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, img):
        skips = []

        for encoder in self.encoders:
            skip, img = encoder(img)
            skips.append(skip)

        img = self.bottleneck(img)


        for decoder, skip in zip(self.decoders, reversed(skips)):
            img = decoder(img, skip)

        img = self.final_conv(img)

        return img

if __name__ == '__main__':
    unet = UNet()

    from PIL import Image

    FILEPATH = "./data/pokemon/1.png"
    OUTFILE = "./test.png"

    img = Image.open(FILEPATH)
    img = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])(img)

    ret = unet(img)

    import matplotlib.pyplot as plt
    import numpy as np

    ret = ret.detach().numpy()
    ret = np.swapaxes(ret, 0, 1)
    ret = np.swapaxes(ret, 1, 2)


    plt.imsave(OUTFILE, np.clip(ret, 0, 1))
