from unet import UNet

import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms as T

from tqdm import tqdm
from icecream import ic

import wandb

EPOCHS = 100
BATCH_SIZE = 128

LR = 1e-4

def train():
    run = wandb.init(project='UNetAutoEncoder', entity='stable-diff-ramit-baily')

    unet = UNet()

    DATAPATH = "./data/"
    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])

    data_set = ImageFolder(DATAPATH, transform=transform)
    data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = Adam(unet.parameters(), lr=LR)
    criterion = MSELoss()

    with tqdm(range(EPOCHS)) as epochs:
        for epoch in epochs:
            avg_loss = 0

            for i, imgs in enumerate(data_loader):
                imgs = imgs[0]

                outputs = unet(imgs)

                loss = criterion(outputs, imgs)
                avg_loss += loss.item()
                loss.backward()

                optimizer.step()

                if i == 0:
                    img = imgs[0]
                    output = outputs[0]

                    run.log({
                        'Real' : wandb.Image(img),
                        'Generated' : wandb.Image(output)
                    })
            
            epochs.set_postfix({
                'Loss' : avg_loss / len(data_loader),
                'Epoch' : epoch
            })

            run.log({
                'Loss' : avg_loss / len(data_loader),
                'Epoch' : epoch
            })
    
    run.finish()

if __name__ == '__main__':
    train()