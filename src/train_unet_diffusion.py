from modules.unet import UNet
from diffusion_utils import add_gaussian_noise
from modules.lpips import LPIPS

from tqdm import tqdm

from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader

import torch

from icecream import ic
import wandb

# =========================================
#     Data Structures Used
# =========================================

class dotdict(dict):
    """
    Dot notation to access dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

config = dotdict({
    'features' : (16, 32, 64),
    'lr' : 1e-5,
    'epochs' : 15,
    'image_shape' : (1, 3, 128, 128),

    'data_source' : 'data_128x128/',

    'diffusion_rounds' : 100,
    'diffusion_mean' : 0,
    'diffusion_std' : 0.001,
    'batch_size' : 16,

    'loss_type' : 'Huber'
})

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class UNetDiffusionTrainer:
    def __init__(self, config, device):
        self.device = device

        self.run = wandb.init(project='UNet Diffusion', entity='stable-diff-ramit-baily', config=config)
        self.config = self.run.config

        self.model = UNet(features = self.config.features).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = config.lr)

        if self.config.loss_type == 'Binary Crossentropy': # FIXME generates NaN loss
            self.criterion = torch.nn.BCELoss()
        elif self.config.loss_type == 'Mean Squared Error':
            self.criterion = torch.nn.MSELoss()
        elif self.config.loss_type == 'Huber':
            self.criterion = torch.nn.HuberLoss()
        elif self.config.loss_type == 'Perceptual':
            self.criterion = LPIPS().eval().to(device)
        else:
            raise Exception("Invalid Loss Type")

    def train(self):
        transform  = T.Compose([
            T.PILToTensor(),
            T.ConvertImageDtype(torch.float)
        ])

        data_set = ImageFolder(self.config.data_source, transform=transform)
        data_loader = DataLoader(data_set, batch_size=self.config.batch_size, shuffle=True)

        with tqdm(range(self.config.epochs)) as epochs:
            for epoch in epochs:
                for i, imgs in enumerate(data_loader):
                    imgs = imgs[0].to(device)

                    avg_loss = 0
                    with tqdm(range(self.config.diffusion_rounds), 
                        desc=f'Training Diffusion Model (Batch {i + 1} / {len(data_loader)})', leave=False) as training_pbar:
                        for _ in training_pbar:
                            noised_imgs = add_gaussian_noise(
                                imgs, mean=self.config.diffusion_mean, 
                                std=self.config.diffusion_std,
                                device = self.device
                            )
                        
                            self.optimizer.zero_grad()

                            outputs = self.model(noised_imgs)
                            loss = self.criterion(imgs, noised_imgs - outputs).mean()
                            avg_loss += loss.item()
                            loss.backward()
                            self.optimizer.step()

                            imgs = noised_imgs
                        
                            training_pbar.set_postfix(loss = loss.item())
                            self.run.log({
                                'loss' : loss.item()
                            })

                    epochs.set_postfix(
                        batch_loss = avg_loss / self.config.diffusion_rounds
                    )

                    self.run.log({
                        'batch_loss' : avg_loss / self.config.diffusion_rounds
                    }, commit = False)

                    # ic()

                    with torch.no_grad():

                        test_random = torch.clip(
                            torch.normal(mean = 0.5, std = 0.5, size=self.config.image_shape),
                            min = 0, max = 1).to(self.device)

                        for i in range(self.config.diffusion_rounds):
                            test_random = test_random - self.model(test_random)

                        self.run.log({'Generated Image' : wandb.Image(test_random)})

                    # ic()

def main():
    trainer = UNetDiffusionTrainer(config, device)
    trainer.train()

if __name__ == '__main__':
    main()