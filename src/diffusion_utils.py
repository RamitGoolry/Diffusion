import torch
import torchvision.transforms as T
from tqdm import tqdm

from icecream import ic

def add_gaussian_noise(img, mean, std):
    return img + torch.normal(mean, std, size=img.shape)

if __name__ == '__main__':
    from PIL import Image

    FILEPATH = "./data/pokemon/2.png"
    OUTFILE = "./test.png"

    img = Image.open(FILEPATH)
    img = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])(img)

    for i in tqdm(range(1)):
        img = add_gaussian_noise(img, 0, 0.01)

    img = T.ToPILImage()(img)

    img.save(OUTFILE)