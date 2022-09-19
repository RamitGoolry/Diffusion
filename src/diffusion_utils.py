import torch
import torchvision.transforms as T
from tqdm import tqdm

def add_gaussian_noise(img, mean, std):
    return img + torch.normal(mean, std, size=img.shape)

if __name__ == '__main__':
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    FILEPATH = "./data/pokemon/2.png"
    OUTFILE = "./test.png"

    img = Image.open(FILEPATH)
    img = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])(img)

    for i in tqdm(range(50)):
        img = add_gaussian_noise(img, 0, 0.05)

    img = np.array(img)
    img = np.swapaxes(img, 0, 1)
    img = np.swapaxes(img, 1, 2)

    plt.imsave(OUTFILE, np.clip(img, 0, 1))