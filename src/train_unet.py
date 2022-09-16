from modules.unet import UNet

import torch
from torch.optim import Adam
from torch.nn import MSELoss, BCELoss
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms as T

from tqdm import tqdm

import wandb
from torchsummary import summary

config = {
    'epochs' : 1000,
    'batch_size' : 32,
    'lr' : 1e-4,
    'features' : (16, 32, 64)
}

def train():
    run = wandb.init(
            project='UNetAutoEncoder',
            entity='stable-diff-ramit-baily',
            config = config,
        )

    unet = UNet(features=run.config.features)
    unet = unet.cuda()

    print(summary(unet, input_size=(3, 256, 256)))

    DATAPATH = "./data/"
    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])

    data_set = ImageFolder(DATAPATH, transform=transform)
    data_loader = DataLoader(data_set, batch_size=run.config.batch_size, shuffle=True)

    optimizer = Adam(unet.parameters(), lr=run.config.lr)
    criterion = BCELoss()

    with tqdm(range(run.config.epochs)) as epochs:
        for epoch in epochs:
            avg_loss = 0

            for i, imgs in enumerate(data_loader):
                imgs = imgs[0]
                imgs = imgs.cuda()

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
