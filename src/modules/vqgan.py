import torch
import torch.nn as nn
from .vqgan_helpers import ResidualBlock, NonLocalBlock, DownSampleBlock, UpSampleBlock, GroupNorm, Swish

from icecream import ic
from torchsummary import summary

class VQGANEncoder(nn.Module):
    def __init__(self, config):
        super(VQGANEncoder, self).__init__()

        channels = [128, 256, 256] # TODO Parameterize
        attn_resolutions = [16]

        num_res_blocks = 2
        resolution = 256 # TODO remove hard code -> Parameterize

        layers = [nn.Conv2d(config.image_channels, channels[0], 3, 1, 1)]

        for i in range(len(channels) - 1):
            in_channels = channels[i]
            out_channels = channels[i + 1]

            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))

            if i != len(channels) - 2:
                layers.append(DownSampleBlock(channels[i + 1]))
                resolution //= 2

        layers.extend([
            ResidualBlock(channels[-1], channels[-1]),
            NonLocalBlock(channels[-1]),
            ResidualBlock(channels[-1], channels[-1]),
            GroupNorm(channels[-1]),
            Swish(),
            nn.Conv2d(channels[-1], config.latent_dim, 3, 1, 1)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class VQGANDecoder(nn.Module):
    def __init__(self, config):
        super(VQGANDecoder, self).__init__()

        channels = [256, 256, 128] # TODO Parameterize
        attn_resolutions = [16]

        num_res_blocks = 2
        resolution = 16

        in_channels = channels[0]
        layers = [
            nn.Conv2d(config.latent_dim, in_channels, 3, 1, 1),
            ResidualBlock(in_channels, in_channels),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels)
        ]

        for i in range(len(channels)):
            out_channels = channels[i]

            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels

                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))

            if i > 1:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2

        layers.extend([
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, config.image_channels, 3, 1, 1)
        ])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Codebook(nn.Module):
    def __init__(self, config):
        super(Codebook, self).__init__()

        self.num_codebook_vectors = config.num_codebook_vectors
        self.latent_dim = config.latent_dim
        self.beta = config.beta

        self.embedding = nn.Embedding(self.num_codebook_vectors, self.latent_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_codebook_vectors, 1.0 / self.num_codebook_vectors)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.latent_dim)

        d = torch.sum(z_flattened ** 2, dim = 1, keepdim = True) + \
            torch.sum(self.embedding.weight ** 2, dim = 1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
            # (a - b)^2 = a^2 + b^2 - 2ab
        
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
         # NOTE .detach() is essentially the stop-gradient operation : It fixes the gradients such that they can not be learned
         # from on the iteration.

        z_q = z + (z_q - z).detach() # For Gradients, otherwise this is a no-op
        z_q = z_q.permute(0, 3, 1, 2)

        return z_q, min_encoding_indices, loss

class Discriminator(nn.Module):
    def __init__(self, config, num_filters_last = 64, n_layers = 3):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv2d(config.image_channels, num_filters_last, 4, 2, 1),
            nn.LeakyReLU(0.2)
        ]

        num_filters_mult = 1

        for i in range(1, n_layers + 1):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)

            layers += [
                nn.Conv2d(num_filters_last * num_filters_mult_last, num_filters_last * num_filters_mult, 
                    4, 2 if i < n_layers else 1, 1, bias=False),
                # nn.BatchNorm2d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        layers.append(
            nn.Conv2d(num_filters_last * num_filters_mult, 1, 4, 1, 1)
        )
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class VQGAN(nn.Module):
    def __init__(self, config, device):
        super(VQGAN, self).__init__()

        self.encoder = VQGANEncoder(config).bfloat16().to(device=device)
        self.decoder = VQGANDecoder(config).bfloat16().to(device=device)
        self.codebook = Codebook(config).bfloat16().to(device=device)

        self.quant_conv = nn.Conv2d(config.latent_dim, config.latent_dim, 1).bfloat16().to(device=device)
        self.post_quant_conv = nn.Conv2d(config.latent_dim, config.latent_dim, 1).bfloat16().to(device=device)

    def forward(self, imgs):
        encoded_imgs = self.encoder(imgs)
        quant_conv_encoded_imgs = self.quant_conv(encoded_imgs)

        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_imgs)

        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_images = self.decoder(post_quant_conv_mapping)

        return decoded_images, codebook_indices, q_loss
    
    def encode(self, imgs):
        encoded_imgs = self.encoder(imgs)
        quant_conv_encoded_imgs = self.quant_conv(encoded_imgs)

        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_imgs)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z):
        post_quant_conv_mapping = self.post_quant_conv(z)
        decoded_images = self.decoder(post_quant_conv_mapping)
        return decoded_images

    DELTA = 1e-6
    
    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.model[-1]

        # TODO Check if there is a need to create a new var "last_layer_weight" here
        # Only issue would be if there is a deep copy that changes the weights

        perceptual_loss_grads = torch.autograd.grad(
            perceptual_loss, last_layer.weight, retain_graph=True
        )[0]
        gan_loss_grads = torch.autograd.grad(
            gan_loss, last_layer.weight, retain_graph=True
        )[0]

        ?? = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + VQGAN.DELTA)
        ?? = torch.clamp(??, 0, 1e4).detach()

        return 0.8 * ??
    
    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))