from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F

from modules.lpips import LPIPS
from modules.vqgan import VQGAN, Discriminator
from modules.vqgan_utils import load_data, weights_init

from icecream import ic

class dotdict(dict):
    """
    Dot notation to access dictionary attributes
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

config = dotdict({
    'latent_dim' : 256,              # Latent Dimension
    'image_size' : 256,              # Image Size
    "num_codebook_vectors" : 1024,   # Number of Codebook Vectors
    "beta" : 0.25,                   # 
    "image_channels" : 3,            # Number of Image Channels
    "dataset_path" : "data/pokemon", # Path to the dataset
    "batch_size" : 1,                # Batch Size
    "epochs" : 100,                  # Epochs
    "learning_rate" : 2.25e-5,       # Learning Rate for both optimizers
    "beta1" : 0.5,                   # Adam beta1
    "beta2" : 0.999,                 # Adam beta2
    "disc_start" : 10000,            # Step at which discriminator will start
    "disc_factor" : 1.,              # Weight of discriminator
    "rec_loss_factor" : 1.,          # Reconstruction Loss weight
    "perceptual_loss_factor" : 1.    # Perceptual Loss Weight
})

class VQGAN_Trainer:
    def __init__(self, config, device):
        self.device = device

        self.vqgan = VQGAN(config, device).to(device)
        self.discriminator = Discriminator(config).bfloat16().to(device)
        self.discriminator.apply(weights_init)

        self.perceptual_loss = LPIPS().eval().to(device)
        self.opt_vq, self.opt_disc = self.configure_optimizers(config)

        self.config = config

    def configure_optimizers(self, config):
        opt_vq = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()) +
            list(self.vqgan.decoder.parameters()) +
            list(self.vqgan.codebook.parameters()) +
            list(self.vqgan.quant_conv.parameters()) +
            list(self.vqgan.post_quant_conv.parameters()),
            lr=config.learning_rate, eps=1e-8, betas = (config.beta1, config.beta2)
        )

        opt_disc = torch.optim.Adam(self.discriminator.parameters(),
            lr=config.learning_rate, eps=1e-8, betas = (config.beta1, config.beta2)
        )

        return opt_vq, opt_disc

    def train(self):
        train_dataset = load_data(config)
        steps_per_epoch = len(train_dataset)

        with tqdm(range(config.epochs)) as pbar:
            for epoch in pbar:
                avg_vq_loss = 0
                avg_gan_loss = 0

                for i, imgs in tqdm(enumerate(train_dataset), total=len(train_dataset), leave=False):
                    imgs = imgs.bfloat16().to(self.device)

                    decoded_imgs, _, q_loss = self.vqgan(imgs)

                    disc_real = self.discriminator(imgs)
                    disc_fake = self.discriminator(decoded_imgs)

                    disc_factor = self.vqgan.adopt_weight(
                        self.config.disc_factor, epoch * steps_per_epoch + i,
                        threshold=self.config.disc_start
                    )

                    perceptual_loss = self.perceptual_loss(imgs, decoded_imgs)
                    rec_loss = torch.abs(imgs - decoded_imgs)

                    perceptual_rec_loss = self.config.perceptual_loss_factor * perceptual_loss + \
                        self.config.rec_loss_factor * rec_loss

                    perceptual_rec_loss = perceptual_rec_loss.mean()
                    g_loss = -torch.mean(disc_fake)

                    λ = self.vqgan.calculate_lambda(perceptual_rec_loss, g_loss)
                    vq_loss = perceptual_rec_loss + q_loss + disc_factor * λ * g_loss

                    d_loss_real = torch.mean(F.relu(1. - disc_real))
                    d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                    self.opt_vq.zero_grad()
                    vq_loss.backward(retain_graph = True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vq.step()
                    self.opt_disc.step()

                    avg_vq_loss += vq_loss.cpu().detach().type(torch.float32).numpy()
                    avg_gan_loss += gan_loss.cpu().detach().type(torch.float32).numpy()

                pbar.set_postfix(
                    VQ_Loss = avg_vq_loss / len(train_dataset),
                    GAN_Loss = avg_gan_loss / len(train_dataset)
                )

def main():
    trainer = VQGAN_Trainer(config, device)
    trainer.train()

if __name__ == '__main__':
    main()